"""
pinn_inverse.py
===============
Joint inverse PINN for stiffness-damage identification (10-DOF shear building).

Reuses the actual root-level pinn_model.py API:
    calculate_response_derivatives, calculate_physics_residual,
    derivatives_in_batches, rel_l2, load_trained_model, prepare_training_data

This module does not change or replace structure.py. The driver creates cases using
its existing datasets/simulator interface, then passes the physical case to
prepare_inverse_data().

Nothing here is executed at import time.

Formulation (all in the project's scaled units, tau = t/T0, x~ = x/X0):
    M~ a~ + C~ v~ + K~(alpha) x~ = F~
    K~(alpha) = (T0^2 / M0) * sum_i k_i (1 - alpha_i) E_i
    alpha_i   = ALPHA_MAX * sigmoid(alpha_raw_i)
Damping C does not depend on alpha (verified earlier: C_damaged == C_healthy).
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf
from tensorflow import keras

# This project keeps the forward PINN utilities in the root-level pinn_model.py.
# The fallback also supports importing this file as part of a package.
from models1 import (
        calculate_physics_residual,
        calculate_response_derivatives,
        derivatives_in_batches,
        load_trained_model,
        prepare_training_data,
        rel_l2,
)


F64 = tf.float64
ALPHA_MAX = 0.6


# ============================================================
# 1. DAMAGE PARAMETERS AND STIFFNESS ASSEMBLY
# ============================================================

def element_matrices(n: int) -> np.ndarray:
    """Return element matrices matching StructuralDynamics' fixed convention.

    Element 0 connects the ground to DOF 0; element i>0 connects DOFs i-1 and i.
    This helper intentionally lives here because the inverse stiffness must be
    assembled with TensorFlow operations so gradients can reach alpha_raw.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("n must be a positive integer.")
    E = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[i] = 1.0
        if i > 0:
            e[i - 1] = -1.0
        E[i] = np.outer(e, e)
    return E


def stiffness_physical(alpha: np.ndarray, k_healthy: np.ndarray) -> np.ndarray:
    """Reference stiffness matrix in physical units using the fixed element convention."""
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    k_healthy = np.asarray(k_healthy, dtype=np.float64).reshape(-1)
    if alpha.shape != k_healthy.shape:
        raise ValueError("alpha and k_healthy must have the same one-dimensional shape.")
    if not np.all(np.isfinite(alpha)) or np.any((alpha < 0.0) | (alpha >= 1.0)):
        raise ValueError("alpha entries must be finite and in [0, 1).")
    if not np.all(np.isfinite(k_healthy)) or np.any(k_healthy <= 0.0):
        raise ValueError("Healthy element stiffnesses must be positive and finite.")
    return np.einsum("i,ijk->jk", k_healthy * (1.0 - alpha), element_matrices(len(k_healthy)))


def true_alpha(n: int, elements, severities) -> np.ndarray:
    """Build a ground-truth damage vector for evaluation only (zero-based indices)."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("n must be a positive integer.")
    elements = list(elements)
    severities = list(severities)
    if len(elements) != len(severities):
        raise ValueError("elements and severities must have the same length.")
    if len(set(elements)) != len(elements):
        raise ValueError("Duplicate damaged elements are not allowed.")
    alpha = np.zeros(n, dtype=np.float64)
    for element, severity in zip(elements, severities):
        if not isinstance(element, (int, np.integer)) or isinstance(element, (bool, np.bool_)):
            raise ValueError("Each damage element must be an integer.")
        if not 0 <= int(element) < n:
            raise ValueError(f"Damage element must be between 0 and {n - 1}.")
        severity = float(severity)
        if not np.isfinite(severity) or not 0.0 <= severity < 1.0:
            raise ValueError("Damage severity must be finite and in [0, 1).")
        alpha[int(element)] = severity
    return alpha


def load_pretrained_checkpoint(checkpoint_path: str | Path) -> tuple[keras.Model, dict[str, float]]:
    """Load the existing Fourier PINN and require its matching scale sidecar."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
    scale_path = path.with_name(f"{path.stem}_scales.json")
    if not scale_path.is_file():
        raise FileNotFoundError(f"Required checkpoint scale sidecar not found: {scale_path}")
    try:
        raw_scales = json.loads(scale_path.read_text(encoding="utf-8"))
        scales = {key: float(raw_scales[key]) for key in ("T0", "M0", "X0")}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scale metadata in {scale_path}") from exc
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError(f"T0, M0 and X0 must be positive and finite in {scale_path}")
    model = load_trained_model(path)
    model._pinn_scales = scales.copy()
    return model, scales


def prepare_inverse_data(case, scales: Mapping[str, float]) -> dict[str, Any]:
    """Scale a physical simulation case with healthy-checkpoint scales and hide K.

    The case must follow datasets.py's output schema. The inverse loss will use
    estimated K(alpha), never the simulator's true damaged stiffness matrix.
    """
    P = prepare_training_data(
        case=case,
        K_ref=None,
        scales=scales,
        dtype=F64,
        expose_K=False,
    )
    if "K" in P or "stiffness_matrix" in P:
        raise RuntimeError("Inverse data preparation exposed the true stiffness matrix.")
    return P


def verify_structure_assembly(structural_system, tol: float = 1e-10) -> dict[str, float]:
    """Check the inverse element assembly against the fixed StructuralDynamics API.

    Pass an instance of the existing structure.StructuralDynamics. No changes to
    structure.py are needed. Tests both healthy K and one deterministic damaged K.
    """
    k = np.asarray(structural_system.stiffness, dtype=np.float64).reshape(-1)
    K_from_class = np.asarray(structural_system.build_stiffness_matrix(), dtype=np.float64)
    K_from_elements = stiffness_physical(np.zeros_like(k), k)
    healthy_rel = float(np.linalg.norm(K_from_elements - K_from_class) /
                        max(np.linalg.norm(K_from_class), 1e-12))
    if healthy_rel > tol:
        raise ValueError(f"Inverse stiffness assembly disagrees with StructuralDynamics: {healthy_rel:.3e}")

    # Check a nonuniform alpha vector without modifying the supplied object.
    alpha_test = np.zeros_like(k)
    alpha_test[min(1, len(k) - 1)] = 0.17
    cls_type = type(structural_system)
    test_system = cls_type(
        masses=structural_system.masses.copy(),
        stiffness=(k * (1.0 - alpha_test)),
        damping=structural_system.damping.copy(),
    )
    K_class_damaged = np.asarray(test_system.build_stiffness_matrix(), dtype=np.float64)
    K_element_damaged = stiffness_physical(alpha_test, k)
    damaged_rel = float(np.linalg.norm(K_element_damaged - K_class_damaged) /
                         max(np.linalg.norm(K_class_damaged), 1e-12))
    if damaged_rel > tol:
        raise ValueError(f"Damaged inverse stiffness assembly disagrees with StructuralDynamics: {damaged_rel:.3e}")
    return {"healthy_stiffness_relative_error": healthy_rel,
            "damaged_stiffness_relative_error": damaged_rel}


class DamageParameters:
    """Trainable, bounded damage vector and the differentiable scaled K(alpha)."""

    def __init__(self, k_healthy, T0: float, M0: float,
                 alpha_init: float = 0.02, alpha_max: float = ALPHA_MAX):
        k_healthy = np.asarray(k_healthy, dtype=np.float64).reshape(-1)
        if k_healthy.size < 1 or not np.all(np.isfinite(k_healthy)) or np.any(k_healthy <= 0.0):
            raise ValueError("k_healthy must be a non-empty vector of positive finite stiffnesses.")
        if not (np.isfinite(alpha_init) and np.isfinite(alpha_max) and 0.0 < alpha_init < alpha_max < 1.0):
            raise ValueError("Require 0 < alpha_init < alpha_max < 1, both finite.")
        if not (np.isfinite(T0) and np.isfinite(M0) and T0 > 0.0 and M0 > 0.0):
            raise ValueError("T0 and M0 must be positive and finite.")
        self.n = len(k_healthy)
        self.alpha_max = float(alpha_max)
        self.k_healthy = tf.constant(k_healthy, F64)
        self.E = tf.constant(element_matrices(self.n), F64)
        self.scale = float(T0) ** 2 / float(M0)          # same scaling as prepare_training_data
        p = alpha_init / alpha_max
        raw0 = np.log(p / (1.0 - p))                      # inverse sigmoid
        self.alpha_raw = tf.Variable(np.full(self.n, raw0), dtype=F64, name="alpha_raw")

    def alpha(self) -> tf.Tensor:
        return self.alpha_max * tf.sigmoid(self.alpha_raw)

    def stiffness(self) -> tf.Tensor:
        """Scaled K(alpha); fully differentiable w.r.t. alpha_raw."""
        k_el = self.k_healthy * (1.0 - self.alpha())
        return tf.einsum("i,ijk->jk", k_el, self.E) * self.scale


# ============================================================
# 2. CONFIG
# ============================================================

@dataclass
class InverseConfig:
    seed: int = 42
    lam_data: float = 1.0
    lam_phys: float = 1e-2
    lam_ic: float = 1.0
    n_col: int = 1024
    data_stride: int = 4              # every 4th time point, all DOFs (same as healthy stage)
    adapt_steps: int = 5000           # data+IC only, alpha frozen (network adapts to damaged data)
    joint_steps: int = 12000          # network + alpha, physics on
    lr_net_adapt: float = 1e-3
    lr_net_joint: float = 3e-4
    lr_alpha: float = 1e-2            # NOT validated: treat as a starting value
    lr_decay_net: float = 0.05        # lr *= decay ** (step/steps) inside each phase
    lr_decay_alpha: float = 0.2
    clipnorm_net: float = 1.0
    alpha_init: float = 0.02
    alpha_max: float = ALPHA_MAX
    log_every: int = 500
    loc_threshold: float = 0.05       # for counting false positives only


# ============================================================
# 3. LOSSES
# ============================================================

def inverse_losses(model, dmg: DamageParameters, P: Mapping[str, tf.Tensor],
                   idx_data, w: Mapping[str, Any], n_col: int,
                   include_physics: bool = True):
    """Return (total, L_data, L_phys, L_ic). P must NOT contain the true K."""
    tau, x = P["tau"], P["x"]
    M, C, F = P["M"], P["C"], P["F"]

    td = tf.gather(tau, idx_data)
    xd = tf.gather(x, idx_data)
    L_data = tf.reduce_mean(tf.square(model(td) - xd))

    z = tf.zeros((1, 1), F64)
    x0, v0, _ = calculate_response_derivatives(model, z)
    L_ic = tf.reduce_mean(tf.square(x0)) + tf.reduce_mean(tf.square(v0))

    if include_physics:
        K = dmg.stiffness()                                        # differentiable in alpha_raw
        n_t = tf.shape(tau)[0]
        ci = tf.random.uniform([n_col], 0, n_t, dtype=tf.int32)
        tc, Fc = tf.gather(tau, ci), tf.gather(F, ci)
        _, _, _, R = calculate_physics_residual(model, tc, M, C, K, Fc)
        f_scale = tf.maximum(tf.reduce_mean(tf.square(F)), tf.constant(1e-12, F64))
        L_phys = tf.reduce_mean(tf.square(R)) / f_scale             # same definition as healthy stage
    else:
        L_phys = tf.constant(0.0, F64)

    total = (tf.cast(w["data"], F64) * L_data
             + tf.cast(w["phys"], F64) * L_phys
             + tf.cast(w["ic"], F64) * L_ic)
    return total, L_data, L_phys, L_ic


# ============================================================
# 4. EVALUATION
# ============================================================

def _np(P, key):
    v = P[key]
    return np.asarray(v.numpy() if tf.is_tensor(v) else v, dtype=np.float64)


def damage_report(alpha_hat, alpha_true, loc_threshold: float = 0.05) -> dict:
    alpha_hat = np.asarray(alpha_hat, float)
    alpha_true = np.asarray(alpha_true, float)
    true_idx = np.flatnonzero(alpha_true > 0)
    healthy = np.flatnonzero(alpha_true == 0)
    order = np.argsort(-alpha_hat)                       # descending severity
    rank = {int(i): int(np.where(order == i)[0][0]) + 1 for i in true_idx}
    return {
        "alpha_hat": alpha_hat.tolist(),
        "alpha_true": alpha_true.tolist(),
        "true_elements": true_idx.tolist(),
        "argmax_element": int(order[0]),
        "rank_of_true_elements": rank,
        "alpha_error_at_true": {int(i): float(alpha_hat[i] - alpha_true[i]) for i in true_idx},
        "max_abs_alpha_error": float(np.max(np.abs(alpha_hat - alpha_true))),
        "max_healthy_alpha": float(np.max(alpha_hat[healthy])) if healthy.size else 0.0,
        "mean_healthy_alpha": float(np.mean(alpha_hat[healthy])) if healthy.size else 0.0,
        "n_healthy_above_threshold": int(np.sum(alpha_hat[healthy] > loc_threshold)),
        "localization_top1_correct": bool(len(true_idx) == 1 and order[0] == true_idx[0]),
    }


def evaluate_inverse(model, dmg: DamageParameters, P, lam_phys: float,
                     lam_ic: float = 1.0, batch_size: int = 512) -> dict:
    """Full-grid deterministic evaluation; simulator v/a are diagnostic only."""
    tau = _np(P, "tau")
    xh, vh, ah = derivatives_in_batches(model, tau, batch_size=batch_size)
    x_obs = _np(P, "x")
    M, C, F = _np(P, "M"), _np(P, "C"), _np(P, "F")
    K = dmg.stiffness().numpy()
    R = ah @ M.T + vh @ C.T + xh @ K.T - F
    f_scale = max(float(np.mean(F ** 2)), 1e-12)
    data_full = float(np.mean((xh - x_obs) ** 2))
    phys_full = float(np.mean(R ** 2) / f_scale)
    # The current experiment assumes zero initial displacement and velocity.
    ic_full = float(np.mean(xh[0] ** 2) + np.mean(vh[0] ** 2))
    return {
        "relL2_x": rel_l2(xh, x_obs),                         # true damaged x == observed x here (clean, full field)
        "relL2_v_diag": rel_l2(vh, _np(P, "v")),              # simulator v/a used for evaluation only
        "relL2_a_diag": rel_l2(ah, _np(P, "a")),
        "data_full": data_full,
        "phys_full": phys_full,
        "ic_full": ic_full,
        "residual_rmse_scaled": float(np.sqrt(np.mean(R ** 2))),
        "residual_max_abs_scaled": float(np.max(np.abs(R))),
        "objective": data_full + float(lam_phys) * phys_full + float(lam_ic) * ic_full,
    }


# ============================================================
# 5. TRAINER
# ============================================================

def _make_step(model, dmg, P, idx_data, w, cfg, joint, opt_net, opt_alpha):
    net_vars = model.trainable_variables

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            total, Ld, Lp, Li = inverse_losses(
                model, dmg, P, idx_data, w, cfg.n_col, include_physics=joint)
        if joint:
            grads = tape.gradient(total, net_vars + [dmg.alpha_raw])
            if any(g is None for g in grads):
                raise RuntimeError("Disconnected gradient in joint step.")
            g_net, g_a = grads[:-1], grads[-1]
            opt_net.apply_gradients(zip(g_net, net_vars))
            opt_alpha.apply_gradients([(g_a, dmg.alpha_raw)])
            return total, Ld, Lp, Li, tf.linalg.global_norm(g_net), tf.norm(g_a)
        g_net = tape.gradient(total, net_vars)
        if any(g is None for g in g_net):
            raise RuntimeError("Disconnected gradient in adaptation step.")
        opt_net.apply_gradients(zip(g_net, net_vars))
        return total, Ld, Lp, Li, tf.linalg.global_norm(g_net), tf.constant(0.0, F64)

    return step


def _save_model(model, path: Path, scales: Mapping[str, float]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    sidecar = path.with_name(f"{path.stem}_scales.json")
    sidecar.write_text(json.dumps({k: float(scales[k]) for k in ("T0", "M0", "X0")}, indent=2))
    return str(path.resolve())


def _save_inverse_checkpoint(model, dmg: DamageParameters, path: Path,
                             scales: Mapping[str, float]) -> dict[str, str]:
    """Save a complete inverse state: neural network, scale metadata, and alpha."""
    model_path = Path(_save_model(model, path, scales))
    alpha = dmg.alpha().numpy().astype(np.float64)
    alpha_path = model_path.with_name(f"{model_path.stem}_alpha.npy")
    alpha_json_path = model_path.with_name(f"{model_path.stem}_alpha.json")
    np.save(alpha_path, alpha)
    alpha_json_path.write_text(json.dumps({
        "alpha": alpha.tolist(),
        "alpha_max": float(dmg.alpha_max),
        "element_indices_are_zero_based": True,
    }, indent=2), encoding="utf-8")
    return {"model": str(model_path), "scales": str(model_path.with_name(f"{model_path.stem}_scales.json")),
            "alpha_npy": str(alpha_path), "alpha_json": str(alpha_json_path)}


def _env_info() -> dict:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
    }


def train_inverse(model, dmg: DamageParameters, P: Mapping[str, tf.Tensor],
                  alpha_true: np.ndarray, cfg: InverseConfig, out_dir,
                  scales: Mapping[str, float], extra_config: Mapping[str, Any] | None = None) -> dict:
    """Run adaptation with alpha frozen, then joint network+damage optimization.

    The caller must load a fresh pretrained model and create a fresh DamageParameters
    object for each lambda comparison. This function intentionally does not generate
    structural simulations or modify structure.py.
    """
    if "K" in P or "stiffness_matrix" in P:
        raise ValueError("P exposes the true stiffness. Prepare it with prepare_inverse_data(case, scales).")
    required = ("tau", "x", "v", "a", "M", "C", "F")
    missing = [key for key in required if key not in P]
    if missing:
        raise KeyError(f"Inverse training data is missing keys: {missing}")

    n_t = int(P["tau"].shape[0])
    n_dof = int(P["x"].shape[1])
    if n_t < 2 or n_dof != dmg.n:
        raise ValueError(f"Incompatible data/damage dimensions: n_time={n_t}, data DOF={n_dof}, alpha size={dmg.n}.")
    if cfg.data_stride < 1 or cfg.n_col < 1 or cfg.log_every < 1:
        raise ValueError("data_stride, n_col, and log_every must be positive.")
    if cfg.adapt_steps < 0 or cfg.joint_steps < 1:
        raise ValueError("adapt_steps must be >= 0 and joint_steps must be >= 1.")
    if not all(key in scales for key in ("T0", "M0", "X0")):
        raise ValueError("scales must contain T0, M0, and X0.")

    alpha_true = np.asarray(alpha_true, dtype=np.float64).reshape(-1)
    if alpha_true.shape != (dmg.n,):
        raise ValueError(f"alpha_true must have shape ({dmg.n},).")
    if not np.all(np.isfinite(alpha_true)) or np.any((alpha_true < 0.0) | (alpha_true >= 1.0)):
        raise ValueError("alpha_true values must be finite and in [0, 1).")

    # This sets collocation/random sampling reproducibly without altering loaded weights.
    tf.keras.utils.set_random_seed(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx_data = tf.constant(np.arange(0, n_t, cfg.data_stride, dtype=np.int32))
    w = {
        "data": tf.Variable(cfg.lam_data, dtype=F64, trainable=False),
        "phys": tf.Variable(0.0, dtype=F64, trainable=False),
        "ic": tf.Variable(cfg.lam_ic, dtype=F64, trainable=False),
    }

    (out / "config.json").write_text(json.dumps(
        {"inverse_config": asdict(cfg), "extra": dict(extra_config or {}),
         "scales": {k: float(scales[k]) for k in ("T0", "M0", "X0")},
         "env": _env_info(), "checkpoint_selection": "full-grid data + lambda_phys*physics + lambda_ic*IC; no alpha_true used"},
        indent=2, default=str), encoding="utf-8")

    hist: dict[str, list] = {k: [] for k in (
        "phase", "step", "elapsed_s", "total", "data", "phys", "ic", "lam_phys",
        "lr_net", "lr_alpha", "grad_net", "grad_alpha_raw", "alpha", "eval")}
    best = {"objective": np.inf, "alpha": None, "alpha_raw": None,
            "model_weights": None, "step": None}
    t_start = time.time()
    global_step = 0

    def run_phase(name: str, steps: int, joint: bool, lr_net0: float, lr_alpha0: float):
        nonlocal global_step
        w["phys"].assign(cfg.lam_phys if joint else 0.0)
        opt_net = keras.optimizers.Adam(lr_net0, clipnorm=cfg.clipnorm_net)
        opt_alpha = keras.optimizers.Adam(lr_alpha0)
        opt_net.build(model.trainable_variables)
        opt_alpha.build([dmg.alpha_raw])
        step_fn = _make_step(model, dmg, P, idx_data, w, cfg, joint, opt_net, opt_alpha)
        print(f"\n{'=' * 78}\n{name} | steps={steps} | lam_phys={float(w['phys'].numpy()):.1e}\n{'=' * 78}")

        for s in range(1, steps + 1):
            frac = s / steps
            lr_n = lr_net0 * cfg.lr_decay_net ** frac
            lr_a = lr_alpha0 * cfg.lr_decay_alpha ** frac
            opt_net.learning_rate.assign(lr_n)
            opt_alpha.learning_rate.assign(lr_a)
            total, Ld, Lp, Li, gn, ga = step_fn()
            global_step += 1

            if s == 1 or s % cfg.log_every == 0 or s == steps:
                ev = evaluate_inverse(model, dmg, P, cfg.lam_phys, cfg.lam_ic)
                a_hat = dmg.alpha().numpy()
                vals = [float(total.numpy()), float(Ld.numpy()), float(Lp.numpy()),
                        float(Li.numpy()), float(gn.numpy()), float(ga.numpy())]
                for key, val in zip(("total", "data", "phys", "ic", "grad_net", "grad_alpha_raw"), vals):
                    hist[key].append(val)
                hist["phase"].append(name)
                hist["step"].append(global_step)
                hist["elapsed_s"].append(time.time() - t_start)
                hist["lam_phys"].append(float(w["phys"].numpy()))
                hist["lr_net"].append(float(lr_n))
                hist["lr_alpha"].append(float(lr_a))
                hist["alpha"].append(a_hat.tolist())
                hist["eval"].append(ev)
                err = float(np.max(np.abs(a_hat - alpha_true)))
                print(f"[{name}] {s:5d}/{steps} | tot={vals[0]:.2e} data={vals[1]:.2e} "
                      f"phys={vals[2]:.2e} ic={vals[3]:.1e} | relL2 x={ev['relL2_x']:.1e} "
                      f"v={ev['relL2_v_diag']:.1e} a={ev['relL2_a_diag']:.1e} | "
                      f"|g_net|={vals[4]:.1e} |g_alpha_raw|={vals[5]:.1e} | max|alpha-alpha*|={err:.3f}")
                if s % (2 * cfg.log_every) == 0 or s == steps:
                    print("    alpha:", np.round(a_hat, 3))
                if not (np.all(np.isfinite(vals)) and np.all(np.isfinite(a_hat)) and
                        all(np.isfinite(float(v)) for v in ev.values())):
                    (out / "history.json").write_text(json.dumps(hist, indent=2, allow_nan=False), encoding="utf-8")
                    raise FloatingPointError(f"Non-finite value at global step {global_step}.")
                # Selection uses observed response/physics/IC only, never alpha_true.
                if joint and ev["objective"] < best["objective"]:
                    best.update(
                        objective=float(ev["objective"]), alpha=a_hat.copy(),
                        alpha_raw=dmg.alpha_raw.numpy().copy(),
                        model_weights=[np.array(v, copy=True) for v in model.get_weights()],
                        step=global_step,
                    )

    if cfg.adapt_steps > 0:
        run_phase("ADAPT: data+IC (alpha frozen)", cfg.adapt_steps, False,
                  cfg.lr_net_adapt, cfg.lr_alpha)
    run_phase("JOINT: network + alpha", cfg.joint_steps, True,
              cfg.lr_net_joint, cfg.lr_alpha)

    # Save the true final state first, including its damage vector.
    final_eval = evaluate_inverse(model, dmg, P, cfg.lam_phys, cfg.lam_ic)
    final_damage = dmg.alpha().numpy().copy()
    final_checkpoint = _save_inverse_checkpoint(model, dmg, out / "inverse_final.keras", scales)

    # Save the best evaluated state and its matching alpha vector as a paired state.
    selected_checkpoint = None
    selected_report = None
    selected_eval = None
    if best["model_weights"] is not None:
        final_model_weights = [np.array(v, copy=True) for v in model.get_weights()]
        final_alpha_raw = dmg.alpha_raw.numpy().copy()
        model.set_weights(best["model_weights"])
        dmg.alpha_raw.assign(best["alpha_raw"])
        selected_eval = evaluate_inverse(model, dmg, P, cfg.lam_phys, cfg.lam_ic)
        selected_checkpoint = _save_inverse_checkpoint(model, dmg, out / "inverse_selected.keras", scales)
        selected_report = damage_report(dmg.alpha().numpy(), alpha_true, cfg.loc_threshold)
        # Restore the final live state; selected state remains saved separately.
        model.set_weights(final_model_weights)
        dmg.alpha_raw.assign(final_alpha_raw)

    results = {
        "final": {"checkpoint": final_checkpoint, "damage": damage_report(final_damage, alpha_true, cfg.loc_threshold),
                  "eval": final_eval},
        "selected": {"checkpoint": selected_checkpoint, "step": best["step"],
                     "objective": (float(best["objective"]) if best["alpha"] is not None else None),
                     "damage": selected_report, "eval": selected_eval},
        "elapsed_s": time.time() - t_start,
    }
    (out / "history.json").write_text(json.dumps(hist, indent=2, allow_nan=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


# ============================================================
# 6. VERIFICATION UTILITIES
# ============================================================

def verify_case_consistency(case_h, case_d, alpha_true, k_healthy, tol: float = 1e-9) -> None:
    """Damaged case really has the intended K; C unchanged; simulator obeys the ODE."""
    Kd = np.asarray(case_d["stiffness_matrix"], float)
    Kp = stiffness_physical(alpha_true, k_healthy)
    err = float(np.linalg.norm(Kp - Kd) / max(float(np.linalg.norm(Kd)), 1e-12))
    if not np.isfinite(err) or err > tol:
        raise ValueError(f"Damaged K does not match K(alpha_true): rel. mismatch {err:.2e}")
    if not np.allclose(case_h["damping_matrix"], case_d["damping_matrix"]):
        raise ValueError("C differs between healthy and damaged cases; residual would need C(alpha).")
    M, C = (np.asarray(case_d[k], float) for k in ("mass_matrix", "damping_matrix"))
    x, v, a, F = (np.asarray(case_d[k], float) for k in ("displacement", "velocity", "acceleration", "force_history"))
    force_norm = max(float(np.linalg.norm(F)), 1e-12)
    res = float(np.linalg.norm(a @ M.T + v @ C.T + x @ Kd.T - F) / force_norm)
    if not np.isfinite(res) or res > 1e-8:
        raise ValueError(f"Simulator residual too large: {res:.2e}")


def verify_checkpoint(model, P_h, x_tol: float = 5e-3) -> dict:
    """Healthy-case sanity check of the loaded pretrained network (scaled units)."""
    xh, vh, ah = derivatives_in_batches(model, _np(P_h, "tau"))
    errs = {"x": rel_l2(xh, _np(P_h, "x")), "v": rel_l2(vh, _np(P_h, "v")), "a": rel_l2(ah, _np(P_h, "a"))}
    print("Pretrained checkpoint on healthy data (relL2):", {k: f"{v:.2e}" for k, v in errs.items()})
    if errs["x"] > x_tol:
        raise ValueError(f"Checkpoint displacement error {errs['x']:.2e} > {x_tol:.0e}; wrong checkpoint or scales?")
    return errs