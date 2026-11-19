"""Validate the structural simulator and least-squares damage recovery.

This script does not train a PINN. It checks:
1. Healthy stiffness-matrix assembly against the simulator.
2. The true-data equation-of-motion residual.
3. Kinematics consistency using numerical derivatives in scaled time.
4. A preliminary stiffness-damage sensitivity proxy for each element.
5. Ideal least-squares alpha recovery for the healthy and three damaged cases.
6. Least-squares sensitivity to perturbations in x, v, and a.

Run from the project root:
    python validate_inverse_readiness.py

Uses the canonical apply_damage() implementation from structure.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from datasets import prepare_healthy_pinn_data, generate_case
from structure import masses, stiffness, damping, forces, apply_damage


# ============================================================
# 0. CONFIGURATION
# ============================================================

PERTURBATION_LEVELS = (1e-3, 1e-2, 5e-2)
DAMAGE_SIGNAL_SEVERITY = 0.10
SENSITIVITY_SEED_BASE = 100
OUTPUT_PATH = Path("models/diagnostics/inverse_readiness_results.json")

DAMAGE_TESTS = (
    {"name": "EASY", "damage_elements": [1], "damage_severities": [0.30]},
    {"name": "MEDIUM", "damage_elements": [5], "damage_severities": [0.20]},
    {"name": "HARD", "damage_elements": [9], "damage_severities": [0.20]},
)

np.set_printoptions(precision=6, suppress=True)


# ============================================================
# 1. SMALL HELPERS
# ============================================================

def damage_to_alpha(n_elements, damage_elements, damage_severities):
    """Build the ground-truth damage vector (zero-based element indices)."""
    if len(damage_elements) != len(damage_severities):
        raise ValueError("Damage elements and severities must have the same length.")

    alpha = np.zeros(n_elements, dtype=np.float64)
    for element, severity in zip(damage_elements, damage_severities):
        if not isinstance(element, (int, np.integer)):
            raise ValueError("Each damage element must be an integer.")
        if not 0 <= element < n_elements:
            raise ValueError(f"Damage element must be between 0 and {n_elements - 1}.")
        if not np.isfinite(severity) or not 0.0 <= severity < 1.0:
            raise ValueError("Damage severity must be finite and in [0, 1).")
        alpha[element] = severity
    return alpha


def relative_l2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def element_matrices(n_elements):
    """Element matrices: element 0 to ground; i>0 connects DOFs i-1 and i."""
    E = np.zeros((n_elements, n_elements, n_elements), dtype=np.float64)
    for i in range(n_elements):
        vector = np.zeros(n_elements, dtype=np.float64)
        vector[i] = 1.0
        if i > 0:
            vector[i - 1] = -1.0
        E[i] = np.outer(vector, vector)
    return E


def assemble_stiffness_matrix(stiffness_vector, E):
    """Assemble K = sum_i k_i E_i from the element-stiffness vector."""
    k = np.asarray(stiffness_vector, dtype=np.float64).reshape(-1)
    if len(k) != E.shape[0]:
        raise ValueError("Stiffness vector length does not match the number of elements.")
    return np.einsum("i,ijk->jk", k, E)


# ============================================================
# 2. SCALING FOR KINEMATICS AND DAMAGE-SIGNAL CHECKS
# ============================================================

def build_scaled_problem(case, K_ref):
    """Scale physical simulation arrays using the healthy reference stiffness."""
    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)
    K = np.asarray(case["stiffness_matrix"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)
    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    t = np.asarray(case["time"], dtype=np.float64).reshape(-1)
    K_ref = np.asarray(K_ref, dtype=np.float64)

    dof = x.shape[1]
    if M.shape != (dof, dof) or C.shape != (dof, dof) or K.shape != (dof, dof):
        raise ValueError("M, C, and K must each have shape (dof, dof).")
    if any(arr.shape != x.shape for arr in (v, a, F)):
        raise ValueError("x, v, a, and force_history must all have shape (n_time, dof).")
    if len(t) != len(x) or len(t) < 5 or not np.all(np.diff(t) > 0):
        raise ValueError("time must be strictly increasing and match the response length.")

    eigenvalues = eigh(K_ref, M, eigvals_only=True)
    if not len(eigenvalues) or not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise ValueError("K_ref and M must produce positive finite eigenvalues.")

    omega_1 = np.sqrt(eigenvalues[0])
    T0 = 1.0 / omega_1
    M0 = float(np.mean(np.diag(M)))
    X0 = float(np.sqrt(np.mean(x**2)))
    if not np.isfinite(X0) or X0 < 1e-12:
        raise ValueError("Reference displacement scale is too small.")
    if not np.isfinite(M0) or M0 <= 0:
        raise ValueError("Reference mass scale must be positive and finite.")

    return {
        "tau": (t / T0).reshape(-1, 1),
        "x": x / X0,
        "v": v * T0 / X0,
        "a": a * T0**2 / X0,
        "M": M / M0,
        "C": C * T0 / M0,
        "K": K * T0**2 / M0,
        "F": F * T0**2 / (M0 * X0),
        "T0": float(T0),
        "M0": M0,
        "X0": X0,
        "tau_max": float(t[-1] / T0),
    }


# ============================================================
# 3. STRUCTURAL CONSISTENCY CHECKS
# ============================================================

def physics_residual_check(case):
    """Check the physical equation M*a + C*v + K*x - F."""
    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)
    K = np.asarray(case["stiffness_matrix"], dtype=np.float64)
    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)

    residual = a @ M.T + v @ C.T + x @ K.T - F
    return {
        "mse": float(np.mean(residual**2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def check_stiffness_matrix(case, stiffness_vector, E, label):
    K_expected = assemble_stiffness_matrix(stiffness_vector, E)
    K_actual = np.asarray(case["stiffness_matrix"], dtype=np.float64)
    mismatch = relative_l2(K_expected, K_actual)
    matches = np.allclose(K_expected, K_actual, rtol=1e-8, atol=1e-8)
    print(f"{label} stiffness matrix match: {matches} (relative mismatch={mismatch:.4e})")
    if not matches:
        raise ValueError(
            f"{label} stiffness matrix does not match element assembly. "
            "Check the actual stiffness vector and element ordering."
        )
    return mismatch


def kinematics_check(P):
    """Compare scaled v with dx/dtau and scaled a with dv/dtau."""
    tau = P["tau"][:, 0]
    v_from_x = np.gradient(P["x"], tau, axis=0, edge_order=2)
    a_from_v = np.gradient(P["v"], tau, axis=0, edge_order=2)
    interior = slice(2, -2)
    return {
        "v_vs_dx": relative_l2(P["v"][interior], v_from_x[interior]),
        "a_vs_dv": relative_l2(P["a"][interior], a_from_v[interior]),
    }


def damage_signal(P, healthy_stiffness, E, damage_severity=0.10):
    """Preliminary sensitivity proxy using the healthy response; not a PINN result."""
    k_healthy = np.asarray(healthy_stiffness, dtype=np.float64).reshape(-1)
    K_healthy = assemble_stiffness_matrix(k_healthy, E)
    signals = []
    force_scale = max(float(np.mean(P["F"]**2)), 1e-12)

    for element in range(len(k_healthy)):
        k_damaged = apply_damage(k_healthy, [element], [damage_severity])
        K_damaged = assemble_stiffness_matrix(k_damaged, E)
        dK_scaled = (K_healthy - K_damaged) * P["T0"]**2 / P["M0"]
        response_change = P["x"] @ dK_scaled.T
        signals.append(float(np.mean(response_change**2) / force_scale))

    return np.asarray(signals, dtype=np.float64)


# ============================================================
# 4. LEAST-SQUARES DAMAGE RECOVERY
# ============================================================

def lstsq_alpha(case, healthy_stiffness):
    """Estimate alpha from full-field simulated displacement, velocity, and acceleration."""
    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)
    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)
    k_healthy = np.asarray(healthy_stiffness, dtype=np.float64).reshape(-1)
    dof = M.shape[0]

    if len(k_healthy) != dof:
        raise ValueError("healthy_stiffness length must match the number of DOFs/elements.")

    E = element_matrices(dof)
    K_healthy = assemble_stiffness_matrix(k_healthy, E)

    # From M*a + C*v + (K_h - sum(alpha_i*k_i*E_i))*x = F:
    # A*alpha = M*a + C*v + K_h*x - F.
    rhs = (a @ M.T + v @ C.T + x @ K_healthy.T - F).reshape(-1)
    A = np.stack(
        [(k_healthy[i] * (x @ E[i].T)).reshape(-1) for i in range(dof)],
        axis=1,
    )
    alpha_hat, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return alpha_hat


def lstsq_sensitivity(case, healthy_stiffness, rel_err=1e-2, seed=0):
    """Perturb x, v, and a with independent Gaussian noise, then estimate alpha."""
    if not np.isfinite(rel_err) or rel_err < 0:
        raise ValueError("rel_err must be finite and non-negative.")

    rng = np.random.default_rng(seed)
    perturbed_case = dict(case)
    for key in ("displacement", "velocity", "acceleration"):
        arr = np.asarray(case[key], dtype=np.float64)
        perturbed_case[key] = arr + rel_err * np.std(arr) * rng.standard_normal(arr.shape)
    return lstsq_alpha(perturbed_case, healthy_stiffness)


def summarize_recovery(case_name, case, alpha_true, healthy_stiffness):
    alpha_hat = lstsq_alpha(case, healthy_stiffness)
    absolute_error = np.abs(alpha_hat - alpha_true)
    true_elements = np.flatnonzero(alpha_true > 0.0).tolist()
    predicted_element = int(np.argmax(alpha_hat))
    correct = (
        predicted_element in true_elements
        if true_elements
        else bool(np.max(np.abs(alpha_hat)) < 1e-3)
    )

    result = {
        "alpha_true": alpha_true.tolist(),
        "alpha_hat": alpha_hat.tolist(),
        "max_abs_error": float(np.max(absolute_error)),
        "mean_abs_error": float(np.mean(absolute_error)),
        "true_elements": true_elements,
        "predicted_element": predicted_element,
        "localization_correct": bool(correct),
    }

    print(f"\n========== LEAST-SQUARES: {case_name} ==========")
    print("True alpha:      ", np.round(alpha_true, 6))
    print("Estimated alpha: ", np.round(alpha_hat, 6))
    print("Absolute error:  ", np.round(absolute_error, 6))
    print(f"Maximum absolute alpha error: {result['max_abs_error']:.6e}")
    print(f"Mean absolute alpha error:    {result['mean_abs_error']:.6e}")
    if true_elements:
        print("True damaged element(s):", true_elements)
        print("Predicted largest-alpha element:", predicted_element)
        print("Localization correct:", correct)
    else:
        print("Healthy false-damage check; max |alpha_hat| should be close to zero.")
    return result


# ============================================================
# 5. RUN DIAGNOSTICS (NO PINN TRAINING)
# ============================================================

def main():
    healthy_stiffness = np.asarray(stiffness, dtype=np.float64).reshape(-1)
    n_elements = len(healthy_stiffness)
    case_healthy = prepare_healthy_pinn_data()
    dof = np.asarray(case_healthy["mass_matrix"]).shape[0]

    if n_elements != dof:
        raise ValueError(
            "This element convention expects one stiffness element per DOF; "
            f"got {n_elements} stiffness entries and {dof} DOFs."
        )

    E = element_matrices(n_elements)

    print("========== INPUT CHECK ==========")
    print("DOF / elements:", dof)
    print("Time samples:", len(case_healthy["time"]))
    print("Healthy stiffness vector:", healthy_stiffness)

    healthy_matrix_mismatch = check_stiffness_matrix(
        case_healthy, healthy_stiffness, E, "Healthy"
    )

    healthy_physics = physics_residual_check(case_healthy)
    print("\n========== TRUE-DATA PHYSICS CHECK (PHYSICAL UNITS) ==========")
    print(f"MSE:     {healthy_physics['mse']:.8e}")
    print(f"RMSE:    {healthy_physics['rmse']:.8e}")
    print(f"Max abs: {healthy_physics['max_abs']:.8e}")

    P_healthy = build_scaled_problem(
        case_healthy,
        K_ref=case_healthy["stiffness_matrix"],
    )

    kin = kinematics_check(P_healthy)
    print("\n========== KINEMATICS CHECK (SCALED TIME) ==========")
    print(f"v versus dx/dtau: {kin['v_vs_dx']:.6e}")
    print(f"a versus dv/dtau: {kin['a_vs_dv']:.6e}")

    signals = damage_signal(
        P_healthy,
        healthy_stiffness,
        E,
        damage_severity=DAMAGE_SIGNAL_SEVERITY,
    )
    print(f"\n========== {DAMAGE_SIGNAL_SEVERITY:.0%} DAMAGE-SIGNAL PROXY ==========")
    for element, value in enumerate(signals):
        print(f"Element {element}: {value:.6e}")

    results = {
        "healthy_stiffness_relative_mismatch": healthy_matrix_mismatch,
        "healthy_physics_residual": healthy_physics,
        "kinematics": kin,
        "damage_signal_severity": DAMAGE_SIGNAL_SEVERITY,
        "damage_signals": signals.tolist(),
        "least_squares": {},
        "sensitivity": {},
    }

    alpha_healthy = damage_to_alpha(n_elements, [], [])
    results["least_squares"]["HEALTHY"] = summarize_recovery(
        "HEALTHY", case_healthy, alpha_healthy, healthy_stiffness
    )

    for case_index, test in enumerate(DAMAGE_TESTS):
        name = test["name"]
        elements = test["damage_elements"]
        severities = test["damage_severities"]

        # Use the single canonical apply_damage() function from structure.py.
        k_damaged_expected = apply_damage(
            healthy_stiffness, elements, severities
        )

        # Generate from original healthy parameters; generate_case applies damage once.
        damaged_case = generate_case(
            masses=masses,
            stiffness=stiffness,
            damping=damping,
            damage_elements=elements,
            damage_severities=severities,
            forces=forces,
        )

        damaged_mismatch = check_stiffness_matrix(
            damaged_case, k_damaged_expected, E, name
        )

        C_healthy = np.asarray(case_healthy["damping_matrix"], dtype=np.float64)
        C_damaged = np.asarray(damaged_case["damping_matrix"], dtype=np.float64)
        damping_unchanged = bool(np.allclose(C_healthy, C_damaged))
        print(f"{name} damping matrix unchanged: {damping_unchanged}")

        alpha_true = damage_to_alpha(n_elements, elements, severities)
        recovery = summarize_recovery(name, damaged_case, alpha_true, healthy_stiffness)
        recovery["stiffness_relative_mismatch"] = damaged_mismatch
        recovery["damping_unchanged"] = damping_unchanged
        results["least_squares"][name] = recovery

        results["sensitivity"][name] = {}
        print(f"\n========== NOISY LEAST-SQUARES: {name} ==========")
        print("True alpha:", np.round(alpha_true, 6))

        for rel_err in PERTURBATION_LEVELS:
            alpha_hat = lstsq_sensitivity(
                damaged_case,
                healthy_stiffness=healthy_stiffness,
                rel_err=rel_err,
                seed=SENSITIVITY_SEED_BASE + case_index,
            )
            abs_error = np.abs(alpha_hat - alpha_true)
            predicted_element = int(np.argmax(alpha_hat))
            item = {
                "alpha_true": alpha_true.tolist(),
                "alpha_hat": alpha_hat.tolist(),
                "max_abs_error": float(np.max(abs_error)),
                "mean_abs_error": float(np.mean(abs_error)),
                "predicted_element": predicted_element,
                "localization_correct": bool(predicted_element in elements),
            }
            results["sensitivity"][name][str(rel_err)] = item

            print(f"\nPerturbation level: {rel_err:g}")
            print("Estimated alpha:", np.round(alpha_hat, 6))
            print("Absolute error: ", np.round(abs_error, 6))
            print(f"Maximum absolute alpha error: {item['max_abs_error']:.6e}")
            print(f"Mean absolute alpha error:    {item['mean_abs_error']:.6e}")
            print("Predicted element:", predicted_element)
            print("Localization correct:", item["localization_correct"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n========== DIAGNOSTICS COMPLETE ==========")
    print("No PINN training was run.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
