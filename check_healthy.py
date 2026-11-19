"""Analyze healthy FourierPINN bias and acceleration error modes/time windows.

Run from the project root after a healthy checkpoint and its *_scales.json
sidecar have been saved. This script does not train a model or overwrite any
checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from datasets import prepare_healthy_pinn_data
from structure import stiffness
from models import predict_response

MODEL_PATH = Path("models/weights/healthy_fourier_pinn.keras")
OUTPUT_PATH = Path("models/diagnostics/healthy_pinn_bias_and_acceleration.json")
PREDICTION_BATCH_SIZE = 256
N_TIME_WINDOWS = 8


def element_matrices(n_elements: int) -> np.ndarray:
    """Return element matrices for the project convention: first element to ground."""
    E = np.zeros((n_elements, n_elements, n_elements), dtype=np.float64)
    for i in range(n_elements):
        direction = np.zeros(n_elements, dtype=np.float64)
        direction[i] = 1.0
        if i > 0:
            direction[i - 1] = -1.0
        E[i] = np.outer(direction, direction)
    return E


def assemble_stiffness(stiffness_vector: np.ndarray, E: np.ndarray) -> np.ndarray:
    return np.einsum("i,ijk->jk", np.asarray(stiffness_vector, dtype=np.float64), E)


def lstsq_alpha(case: dict, healthy_stiffness: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Estimate alpha from physical x/v/a and return alpha, cond(A), rank(A)."""
    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)
    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)
    k0 = np.asarray(healthy_stiffness, dtype=np.float64).reshape(-1)
    n = M.shape[0]
    if k0.shape != (n,):
        raise ValueError(f"healthy_stiffness must have shape ({n},), got {k0.shape}.")
    E = element_matrices(n)
    K0 = assemble_stiffness(k0, E)
    rhs = (a @ M.T + v @ C.T + x @ K0.T - F).reshape(-1)
    A = np.stack([(k0[i] * (x @ E[i].T)).reshape(-1) for i in range(n)], axis=1)
    alpha_hat, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    singular_values = np.linalg.svd(A, compute_uv=False)
    cond_A = float(singular_values[0] / max(singular_values[-1], np.finfo(float).tiny))
    return alpha_hat, cond_A, int(rank)


def rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint not found: {MODEL_PATH}. Train the healthy model first."
        )
    sidecar = MODEL_PATH.with_name(f"{MODEL_PATH.stem}_scales.json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Scale metadata not found: {sidecar}. Physical-unit prediction requires it."
        )

    case_h = prepare_healthy_pinn_data()
    time = np.asarray(case_h["time"], dtype=np.float64).ravel()
    a_true = np.asarray(case_h["acceleration"], dtype=np.float64)
    M = np.asarray(case_h["mass_matrix"], dtype=np.float64)
    K = np.asarray(case_h["stiffness_matrix"], dtype=np.float64)
    k_healthy = np.asarray(stiffness, dtype=np.float64).reshape(-1)

    K_from_elements = assemble_stiffness(k_healthy, element_matrices(len(k_healthy)))
    if not np.allclose(K_from_elements, K, rtol=1e-8, atol=1e-8):
        mismatch = rel_l2(K_from_elements, K)
        raise ValueError(
            f"Healthy stiffness vector/matrix mismatch ({mismatch:.3e}); "
            "verify structure.stiffness and the element convention."
        )

    # --------------------------------------------------------
    # A. False-damage bias from the existing healthy checkpoint
    # --------------------------------------------------------
    x_pred, v_pred, a_pred = predict_response(
        model=MODEL_PATH,
        t=time,
        time_is_physical=True,
        batch_size=PREDICTION_BATCH_SIZE,
    )
    predicted_case = dict(case_h)
    predicted_case["displacement"] = x_pred
    predicted_case["velocity"] = v_pred
    predicted_case["acceleration"] = a_pred

    alpha_hat, cond_A, rank_A = lstsq_alpha(predicted_case, k_healthy)
    print("\n========== A. NETWORK-INDUCED DAMAGE BIAS ==========")
    print("True alpha:", np.zeros_like(alpha_hat))
    print("Estimated alpha:", np.round(alpha_hat, 6))
    print(f"max |alpha_hat|: {np.max(np.abs(alpha_hat)):.6e}")
    print(f"mean |alpha_hat|: {np.mean(np.abs(alpha_hat)):.6e}")
    print(f"Least-squares rank: {rank_A}/{len(k_healthy)}")
    print(f"Condition number of A: {cond_A:.6e}")
    print(f"Healthy-network displacement relative L2: {rel_l2(x_pred, np.asarray(case_h['displacement'])):.6e}")
    print(f"Healthy-network velocity relative L2: {rel_l2(v_pred, np.asarray(case_h['velocity'])):.6e}")
    print(f"Healthy-network acceleration relative L2: {rel_l2(a_pred, a_true):.6e}")
    print("Interpret alpha bias relative to the smallest target damage (0.1).")
    print("Note: do not compare alpha values directly with the residual-signal proxy; they use different scales.")

    # --------------------------------------------------------
    # B. Acceleration error breakdown by structural mode/time
    # --------------------------------------------------------
    error = a_pred - a_true
    omega_squared, Phi = eigh(K, M)
    if np.any(omega_squared <= 0):
        raise ValueError("The healthy generalized eigenproblem has non-positive eigenvalues.")
    # eigh returns M-orthonormal modal vectors. Rows are time samples.
    modal_error = error @ M @ Phi
    modal_true = a_true @ M @ Phi
    modal_error_energy = np.sum(modal_error**2, axis=0)
    total_modal_energy = max(float(np.sum(modal_error_energy)), 1e-30)
    modal_share = modal_error_energy / total_modal_energy

    print("\n========== B1. ACCELERATION ERROR BY MODE ==========")
    modal_rows = []
    for j, eigenvalue in enumerate(omega_squared):
        modal_rel = float(
            np.linalg.norm(modal_error[:, j])
            / max(np.linalg.norm(modal_true[:, j]), 1e-12)
        )
        row = {
            "mode": int(j + 1),
            "omega_rad_s": float(np.sqrt(eigenvalue)),
            "relative_error": modal_rel,
            "share_of_total_squared_modal_error": float(modal_share[j]),
            "error_energy": float(modal_error_energy[j]),
        }
        modal_rows.append(row)
        print(
            f"mode {j + 1:2d} | omega={np.sqrt(eigenvalue):8.3f} rad/s | "
            f"rel err={modal_rel:.3e} | error share={modal_share[j]:.3f}"
        )

    print("\n========== B2. ACCELERATION ERROR BY TIME WINDOW ==========")
    time_rows = []
    for window_i, indices in enumerate(np.array_split(np.arange(len(time)), N_TIME_WINDOWS), start=1):
        e_window = error[indices]
        a_window = a_true[indices]
        window_rel = float(
            np.linalg.norm(e_window) / max(np.linalg.norm(a_window), 1e-12)
        )
        row = {
            "window": int(window_i),
            "start_time_s": float(time[indices[0]]),
            "end_time_s": float(time[indices[-1]]),
            "relative_error": window_rel,
            "error_norm": float(np.linalg.norm(e_window)),
            "true_acceleration_norm": float(np.linalg.norm(a_window)),
        }
        time_rows.append(row)
        print(
            f"window {window_i}: t=[{row['start_time_s']:.3f}, {row['end_time_s']:.3f}] s | "
            f"relative error={window_rel:.3e}"
        )

    out = {
        "checkpoint": str(MODEL_PATH),
        "alpha_bias": {
            "true_alpha": np.zeros_like(alpha_hat).tolist(),
            "estimated_alpha": alpha_hat.tolist(),
            "max_abs_alpha": float(np.max(np.abs(alpha_hat))),
            "mean_abs_alpha": float(np.mean(np.abs(alpha_hat))),
            "least_squares_rank": rank_A,
            "condition_number": cond_A,
        },
        "overall_relative_l2": {
            "displacement": rel_l2(x_pred, np.asarray(case_h["displacement"])),
            "velocity": rel_l2(v_pred, np.asarray(case_h["velocity"])),
            "acceleration": rel_l2(a_pred, a_true),
        },
        "modal_acceleration_error": modal_rows,
        "time_window_acceleration_error": time_rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nAnalysis saved to: {OUTPUT_PATH}")
    print("No model training or checkpoint modification was performed.")


if __name__ == "__main__":
    main()
