"""Reusable Fourier PINN utilities for structural-response learning.

This module defines the model, data preparation, losses, training, saving,
loading, evaluation, and prediction functions. It intentionally does not run
training or prediction when imported. Call train_pinn() / predict_response()
from a separate driver script.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf
from scipy.linalg import eigh
from tensorflow import keras
from tensorflow.keras import layers


_FLOAT_DTYPE = tf.float64
_SCALED_KEYS = ("tau", "x", "v", "a", "M", "C", "K", "F")


# ============================================================
# 1. PREPARE TRAINING DATA: SCALING + TENSOR CONVERSION
# ============================================================

def prepare_training_data(
    case: Mapping[str, Any],
    K_ref: Any | None = None,
    scales: Mapping[str, float] | None = None,
    dtype: tf.dtypes.DType = tf.float64,
    sensor_cols: Sequence[int] | None = None,
    expose_K: bool = True,
) -> dict[str, Any]:
    """Scale structural simulation data and convert it to TensorFlow tensors.

    Parameters
    ----------
    case:
        Physical-unit simulation case.
    K_ref:
        Healthy/reference stiffness matrix in physical units. Required only
        when ``scales`` is None, because it defines the reference time scale.
    scales:
        Previously calculated ``T0``, ``M0``, and ``X0`` values. Reuse the
        healthy-case scales for every damaged case.
    dtype:
        TensorFlow dtype for scaled array values.
    sensor_cols:
        Optional zero-based measured DOF indices. If supplied while calculating
        new scales, ``X0`` is calculated from these columns only. This does not
        itself mask the returned displacement tensor; the inverse trainer must
        restrict its data loss to the observed columns.
    expose_K:
        True returns the case's scaled stiffness matrix as ``K`` and
        ``stiffness_matrix`` (forward/healthy training). False omits both keys,
        to avoid exposing the true stiffness to an inverse-training routine.
        The current forward ``train_pinn`` requires K; an inverse trainer must
        construct and use K(alpha) instead.

    Returns
    -------
    dict
        Scaled TensorFlow tensors plus scale metadata. Canonical keys are
        ``tau``, ``x``, ``v``, ``a``, ``M``, ``C``, ``F`` and, when
        ``expose_K=True``, ``K``. Descriptive aliases are included for backward
        compatibility. All response and matrix arrays are dimensionless/scaled.
    """
    required = (
        "time", "displacement", "velocity", "acceleration",
        "mass_matrix", "damping_matrix", "stiffness_matrix", "force_history",
    )
    missing = [key for key in required if key not in case]
    if missing:
        raise KeyError(f"Simulation case is missing keys: {missing}")

    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)
    K = np.asarray(case["stiffness_matrix"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)
    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    t = np.asarray(case["time"], dtype=np.float64).ravel()

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("mass_matrix must be a square matrix.")
    dof = M.shape[0]
    for name, matrix in (("damping_matrix", C), ("stiffness_matrix", K)):
        if matrix.shape != (dof, dof):
            raise ValueError(f"{name} must have shape {(dof, dof)}.")
    if x.ndim != 2 or x.shape[1] != dof:
        raise ValueError(f"displacement must have shape (n_time, {dof}).")
    n_t = x.shape[0]
    for name, arr in (("velocity", v), ("acceleration", a), ("force_history", F)):
        if arr.shape != (n_t, dof):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(n_t, dof)}.")
    if (
        len(t) != n_t
        or n_t < 2
        or not np.all(np.isfinite(t))
        or abs(t[0]) > 1e-12
        or not np.all(np.diff(t) > 0)
    ):
        raise ValueError(
            "time must start at 0, be finite and strictly increasing, "
            "and match the response length."
        )

    if sensor_cols is not None:
        sensor_cols = list(sensor_cols)
        if not sensor_cols:
            raise ValueError("sensor_cols cannot be empty when supplied.")
        if any(not isinstance(col, (int, np.integer)) or isinstance(col, (bool, np.bool_))
               for col in sensor_cols):
            raise ValueError("sensor_cols must contain integer DOF indices.")
        if len(set(sensor_cols)) != len(sensor_cols):
            raise ValueError("sensor_cols must not contain duplicates.")
        if min(sensor_cols) < 0 or max(sensor_cols) >= dof:
            raise ValueError(f"sensor_cols must be valid indices from 0 to {dof - 1}.")

    if scales is None:
        if K_ref is None:
            raise ValueError("K_ref is required when scales is None.")
        K_ref_array = np.asarray(K_ref, dtype=np.float64)
        if K_ref_array.shape != (dof, dof):
            raise ValueError(f"K_ref must have shape {(dof, dof)}.")

        eigenvalues = eigh(K_ref_array, M, eigvals_only=True)
        if (
            eigenvalues.size == 0
            or not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 0.0
        ):
            raise ValueError("K_ref and M must produce positive finite eigenvalues.")

        omega_1 = float(np.sqrt(eigenvalues[0]))
        T0 = 1.0 / omega_1
        M0 = float(np.mean(np.diag(M)))
        x_ref = x if sensor_cols is None else x[:, sensor_cols]
        X0 = float(np.sqrt(np.mean(x_ref ** 2)))
    else:
        try:
            T0 = float(scales["T0"])
            M0 = float(scales["M0"])
            X0 = float(scales["X0"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scales must contain numeric T0, M0, and X0 values.") from exc

    if not np.isfinite(T0) or T0 <= 0.0:
        raise ValueError("T0 must be positive and finite.")
    if not np.isfinite(M0) or M0 <= 0.0:
        raise ValueError("M0 must be positive and finite.")
    if not np.isfinite(X0) or X0 < 1e-12:
        raise ValueError("X0 must be finite and at least 1e-12.")

    scaled_arrays = {
        "tau": (t / T0).reshape(-1, 1),
        "x": x / X0,
        "v": v * T0 / X0,
        "a": a * T0**2 / X0,
        "M": M / M0,
        "C": C * T0 / M0,
        "F": F * T0**2 / (M0 * X0),
    }
    if expose_K:
        scaled_arrays["K"] = K * T0**2 / M0

    tensor_dtype = tf.as_dtype(dtype)
    output: dict[str, Any] = {
        key: tf.convert_to_tensor(value, dtype=tensor_dtype)
        for key, value in scaled_arrays.items()
    }

    scale_info = {"T0": float(T0), "M0": float(M0), "X0": float(X0)}
    output["scales"] = scale_info.copy()
    output.update(scale_info)
    output["tau_max"] = float(t[-1] / T0)
    output["sensor_cols"] = None if sensor_cols is None else list(sensor_cols)

    # Descriptive aliases preserve names from the original training code.
    aliases = {
        "time": "tau",
        "displacement": "x",
        "velocity": "v",
        "acceleration": "a",
        "mass_matrix": "M",
        "damping_matrix": "C",
        "force_history": "F",
    }
    if expose_K:
        aliases["stiffness_matrix"] = "K"
    for alias, key in aliases.items():
        output[alias] = output[key]

    return output


def _scaled_tensor_dict(P: Mapping[str, Any]) -> dict[str, tf.Tensor]:
    """Return only the canonical scaled tensors needed by training."""
    aliases = {
        "tau": "time",
        "x": "displacement",
        "v": "velocity",
        "a": "acceleration",
        "M": "mass_matrix",
        "C": "damping_matrix",
        "K": "stiffness_matrix",
        "F": "force_history",
    }
    result: dict[str, tf.Tensor] = {}
    for key in _SCALED_KEYS:
        value = P.get(key, P.get(aliases[key]))
        if value is None:
            if key == "K":
                raise KeyError(
                    "Scaled training data has no K matrix. If it was prepared with "
                    "expose_K=False, use an inverse trainer that constructs K(alpha); "
                    "the forward train_pinn() function requires a known K."
                )
            raise KeyError(f"Scaled training data is missing '{key}' (or alias '{aliases[key]}').")
        result[key] = tf.cast(tf.convert_to_tensor(value), _FLOAT_DTYPE)
    return result


def _scaled_numpy(P: Mapping[str, Any], key: str) -> np.ndarray:
    aliases = {
        "tau": "time", "x": "displacement", "v": "velocity",
        "a": "acceleration", "M": "mass_matrix", "C": "damping_matrix",
        "K": "stiffness_matrix", "F": "force_history",
    }
    value = P.get(key, P.get(aliases[key]))
    if value is None:
        raise KeyError(f"Scaled problem is missing '{key}'.")
    if tf.is_tensor(value):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


# ============================================================
# 2. FOURIER PINN
# ============================================================

@keras.utils.register_keras_serializable(package="StructuralPINN")
class FourierPINN(keras.Model):
    """Fourier-feature PINN mapping dimensionless time to DOF responses."""

    def __init__(
        self,
        dof: int,
        tau_max: float,
        w_max: float = 8.0,
        n_freq: int = 56,
        width: int = 128,
        depth: int = 4,
        **kwargs: Any,
    ) -> None:
        # Keep the whole model in float64; otherwise Keras' default model
        # policy can autocast tau to float32 before call() casts it back.
        kwargs.setdefault("dtype", tf.float64)
        super().__init__(**kwargs)
        self.dof = int(dof)
        self._tau_max_value = float(tau_max)
        self.w_max = float(w_max)
        self.n_freq = int(n_freq)
        self.width = int(width)
        self.depth = int(depth)

        if self.dof < 1:
            raise ValueError("dof must be positive.")
        if self._tau_max_value <= 0.0:
            raise ValueError("tau_max must be positive.")
        if self.n_freq < 1 or self.width < 1 or self.depth < 1:
            raise ValueError("n_freq, width, and depth must be positive.")

        self.tau_max = tf.constant(self._tau_max_value, dtype=_FLOAT_DTYPE)
        self.freqs = tf.constant(
            np.linspace(0.25, self.w_max, self.n_freq).reshape(1, -1),
            dtype=_FLOAT_DTYPE,
        )
        self.body = keras.Sequential(
            [
                layers.Dense(self.width, activation="tanh", dtype=_FLOAT_DTYPE)
                for _ in range(self.depth)
            ]
            + [layers.Dense(self.dof, dtype=_FLOAT_DTYPE)],
            name="fourier_pinn_body",
        )

    def call(self, tau: tf.Tensor) -> tf.Tensor:
        tau = tf.cast(tau, _FLOAT_DTYPE)
        arg = tau * self.freqs
        features = tf.concat(
            [tf.sin(arg), tf.cos(arg), tau / self.tau_max],
            axis=1,
        )
        return self.body(features)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dof": self.dof,
                "tau_max": self._tau_max_value,
                "w_max": self.w_max,
                "n_freq": self.n_freq,
                "width": self.width,
                "depth": self.depth,
            }
        )
        return config


def build_model(
    P: Mapping[str, Any],
    dof: int | None = None,
    w_max: float = 8.0,
    n_freq: int = 56,
    width: int = 128,
    depth: int = 4,
) -> FourierPINN:
    """Build and initialize a FourierPINN using the scaled time range."""
    tau = _scaled_numpy(P, "tau")
    if dof is None:
        dof = _scaled_numpy(P, "x").shape[1]
    tau_max = float(P.get("tau_max", tau[-1, 0]))
    model = FourierPINN(
        dof=dof,
        tau_max=tau_max,
        w_max=w_max,
        n_freq=n_freq,
        width=width,
        depth=depth,
    )
    model(tf.zeros((1, 1), dtype=_FLOAT_DTYPE))
    return model


# ============================================================
# 3. RESPONSE DERIVATIVES
# ============================================================

def calculate_response_derivatives(
    model: keras.Model,
    t: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate scaled x, dx/d(tau), and d²x/d(tau)² via autodiff.

    The input ``t`` must be dimensionless time ``tau``. First-gradient
    operations are computed outside tape1's context (but inside tape2's), so
    tape1 does not record its own gradient operations unnecessarily.
    """
    t = tf.cast(tf.convert_to_tensor(t), _FLOAT_DTYPE)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(t)
            x_hat = model(t)

            n_outputs = x_hat.shape[1]
            if n_outputs is None:
                raise ValueError("The model output dimension must be statically known.")

            # Record slicing operations while tape1 is active; otherwise the
            # gradient target x_hat[:, i] may be disconnected from tape1.
            x_columns = [x_hat[:, i:i + 1] for i in range(n_outputs)]

        # Outside tape1's context, but inside tape2's context.
        v_columns = [tape1.gradient(column, t) for column in x_columns]
        if any(column is None for column in v_columns):
            raise RuntimeError("Could not calculate one or more displacement derivatives.")
        v_hat = tf.concat(v_columns, axis=1)

    a_columns = [tape2.gradient(v_columns[i], t) for i in range(n_outputs)]
    if any(column is None for column in a_columns):
        raise RuntimeError("Could not calculate one or more acceleration derivatives.")
    a_hat = tf.concat(a_columns, axis=1)
    del tape1, tape2
    return x_hat, v_hat, a_hat


# ============================================================
# 4. PHYSICS RESIDUAL
# ============================================================

def calculate_physics_residual(
    model: keras.Model,
    t: Any,
    M: Any,
    C: Any,
    K: Any,
    force_batch: Any,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate the dimensionless structural-equation residual."""
    t = tf.cast(tf.convert_to_tensor(t), _FLOAT_DTYPE)
    M = tf.cast(tf.convert_to_tensor(M), _FLOAT_DTYPE)
    C = tf.cast(tf.convert_to_tensor(C), _FLOAT_DTYPE)
    K = tf.cast(tf.convert_to_tensor(K), _FLOAT_DTYPE)
    force_batch = tf.cast(tf.convert_to_tensor(force_batch), _FLOAT_DTYPE)

    x_hat, v_hat, a_hat = calculate_response_derivatives(model, t)
    Ma = tf.matmul(a_hat, M, transpose_b=True)
    Cv = tf.matmul(v_hat, C, transpose_b=True)
    Kx = tf.matmul(x_hat, K, transpose_b=True)
    residual = Ma + Cv + Kx - force_batch
    return x_hat, v_hat, a_hat, residual


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

def calculate_loss_function(
    model: keras.Model,
    P: Mapping[str, Any],
    idx_data: Any,
    w: Mapping[str, Any],
    n_col: int = 1024,
    tau_window: Any | None = None,
    data_only: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate data, physics, and initial-condition losses.

    The data term uses the time indices in idx_data. The physics term uses
    randomly sampled collocation points from the current time window. All
    DOFs are included in the response loss; idx_data subsamples time only.
    """
    P_t = P if all(tf.is_tensor(P.get(k)) for k in _SCALED_KEYS) else _scaled_tensor_dict(P)
    tau_all, x_all = P_t["tau"], P_t["x"]
    M, C, K, F = P_t["M"], P_t["C"], P_t["K"], P_t["F"]
    idx_data = tf.cast(tf.reshape(tf.convert_to_tensor(idx_data), [-1]), tf.int32)

    if tau_window is None:
        tau_window_t = tf.reduce_max(tau_all[:, 0])
    else:
        tau_window_t = tf.cast(tau_window, _FLOAT_DTYPE)

    # Sparse observed response points (all DOFs at each selected time).
    data_mask = tf.gather(tau_all[:, 0] <= tau_window_t, idx_data)
    idx_used = tf.boolean_mask(idx_data, data_mask)
    tf.debugging.assert_positive(
        tf.size(idx_used), message="No observed data points available in tau_window."
    )
    td = tf.gather(tau_all, idx_used)
    xd = tf.gather(x_all, idx_used)
    data_loss = tf.reduce_mean(tf.square(model(td) - xd))

    # Initial conditions: x(0)=0 and v(0)=0.
    t0 = tf.zeros((1, 1), dtype=_FLOAT_DTYPE)
    x0_hat, v0_hat, _ = calculate_response_derivatives(model, t0)
    ic_loss = tf.reduce_mean(tf.square(x0_hat)) + tf.reduce_mean(tf.square(v0_hat))

    # Physics loss can be skipped efficiently during the data-only warm-up.
    if data_only:
        physics_loss = tf.constant(0.0, dtype=_FLOAT_DTYPE)
    else:
        candidate_indices = tf.cast(
            tf.reshape(tf.where(tau_all[:, 0] <= tau_window_t), [-1]), tf.int32
        )
        n_candidates = tf.shape(candidate_indices)[0]
        tf.debugging.assert_positive(
            n_candidates, message="No collocation points available in tau_window."
        )
        local_indices = tf.random.uniform(
            shape=(n_col,), minval=0, maxval=n_candidates, dtype=tf.int32
        )
        collocation_indices = tf.gather(candidate_indices, local_indices)
        tc = tf.gather(tau_all, collocation_indices)
        Fc = tf.gather(F, collocation_indices)
        _, _, _, residual = calculate_physics_residual(model, tc, M, C, K, Fc)
        f_scale = tf.maximum(tf.reduce_mean(tf.square(F)), tf.constant(1e-12, dtype=_FLOAT_DTYPE))
        physics_loss = tf.reduce_mean(tf.square(residual)) / f_scale

    total_loss = (
        tf.cast(w["data"], _FLOAT_DTYPE) * data_loss
        + tf.cast(w["phys"], _FLOAT_DTYPE) * physics_loss
        + tf.cast(w["ic"], _FLOAT_DTYPE) * ic_loss
    )
    return total_loss, data_loss, physics_loss, ic_loss


# ============================================================
# 6. TRAINING STEP
# ============================================================

@tf.function(reduce_retracing=True)
def train_step(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    P: Mapping[str, tf.Tensor],
    idx_data: tf.Tensor,
    w: Mapping[str, tf.Variable],
    n_col: int,
    tau_window: tf.Tensor,
    data_only: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Perform one optimizer update."""
    with tf.GradientTape() as tape:
        total_loss, data_loss, physics_loss, ic_loss = calculate_loss_function(
            model=model,
            P=P,
            idx_data=idx_data,
            w=w,
            n_col=n_col,
            tau_window=tau_window,
            data_only=data_only,
        )

    variables = model.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
    if not pairs:
        raise RuntimeError("No gradients were calculated for model parameters.")
    optimizer.apply_gradients(pairs)
    return total_loss, data_loss, physics_loss, ic_loss


# ============================================================
# 7. EVALUATION UTILITIES
# ============================================================

def derivatives_in_batches(
    model: keras.Model,
    tau: Any,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate scaled displacement, velocity, and acceleration in batches."""
    tau_array = np.asarray(tau.numpy() if tf.is_tensor(tau) else tau, dtype=np.float64)
    if tau_array.ndim == 1:
        tau_array = tau_array.reshape(-1, 1)
    if tau_array.ndim != 2 or tau_array.shape[1] != 1:
        raise ValueError("tau must have shape (n_time,) or (n_time, 1).")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    x_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []
    a_parts: list[np.ndarray] = []
    for start in range(0, len(tau_array), batch_size):
        tau_batch = tf.convert_to_tensor(tau_array[start:start + batch_size], dtype=_FLOAT_DTYPE)
        x_batch, v_batch, a_batch = calculate_response_derivatives(model, tau_batch)
        x_parts.append(x_batch.numpy())
        v_parts.append(v_batch.numpy())
        a_parts.append(a_batch.numpy())

    if not x_parts:
        empty = np.empty((0, getattr(model, "dof", 0)), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(v_parts, axis=0),
        np.concatenate(a_parts, axis=0),
    )


def rel_l2(pred: Any, true: Any) -> float:
    pred_array = np.asarray(pred.numpy() if tf.is_tensor(pred) else pred, dtype=np.float64)
    true_array = np.asarray(true.numpy() if tf.is_tensor(true) else true, dtype=np.float64)
    return float(np.linalg.norm(pred_array - true_array) / max(np.linalg.norm(true_array), 1e-12))


def evaluate_model(model: keras.Model, P: Mapping[str, Any], batch_size: int = 512) -> dict[str, float]:
    tau = _scaled_numpy(P, "tau")
    true_x = _scaled_numpy(P, "x")
    true_v = _scaled_numpy(P, "v")
    true_a = _scaled_numpy(P, "a")
    pred_x, pred_v, pred_a = derivatives_in_batches(model, tau, batch_size=batch_size)
    return {
        "x": rel_l2(pred_x, true_x),
        "v": rel_l2(pred_v, true_v),
        "a": rel_l2(pred_a, true_a),
    }


def get_gradient_norms(
    model: keras.Model,
    P: Mapping[str, Any],
    idx_data: Any,
    n_col: int = 16,
    tau_window: float | None = None,
    seed: int | None = 0,
) -> dict[str, float]:
    """Estimate separate data, physics, and IC gradient norms for diagnostics."""
    P_t = _scaled_tensor_dict(P)
    tau_all, x_all = P_t["tau"], P_t["x"]
    M, C, K, F = P_t["M"], P_t["C"], P_t["K"], P_t["F"]
    if tau_window is None:
        tau_window = float(tf.reduce_max(tau_all[:, 0]).numpy())
    current_window = tf.cast(tau_window, _FLOAT_DTYPE)

    data_indices = tf.cast(tf.reshape(tf.convert_to_tensor(idx_data), [-1]), tf.int32)
    data_mask = tf.gather(tau_all[:, 0] <= current_window, data_indices)
    data_indices = tf.boolean_mask(data_indices, data_mask)
    td = tf.gather(tau_all, data_indices)
    xd = tf.gather(x_all, data_indices)

    with tf.GradientTape() as tape:
        L_data = tf.reduce_mean(tf.square(model(td) - xd))
    grad_data = tape.gradient(L_data, model.trainable_variables)

    with tf.GradientTape() as tape:
        t0 = tf.zeros((1, 1), dtype=_FLOAT_DTYPE)
        x0, v0, _ = calculate_response_derivatives(model, t0)
        L_ic = tf.reduce_mean(tf.square(x0)) + tf.reduce_mean(tf.square(v0))
    grad_ic = tape.gradient(L_ic, model.trainable_variables)

    valid_indices = np.flatnonzero(tau_all[:, 0].numpy() <= float(current_window.numpy()))
    if len(valid_indices) == 0:
        raise ValueError("No collocation points available for gradient diagnostics.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(valid_indices, size=min(n_col, len(valid_indices)), replace=False)
    tc = tf.gather(tau_all, selected)
    Fc = tf.gather(F, selected)
    f_scale = tf.maximum(tf.reduce_mean(tf.square(F)), tf.constant(1e-12, dtype=_FLOAT_DTYPE))

    with tf.GradientTape() as tape:
        _, _, _, residual = calculate_physics_residual(model, tc, M, C, K, Fc)
        L_phys = tf.reduce_mean(tf.square(residual)) / f_scale
    grad_phys = tape.gradient(L_phys, model.trainable_variables)

    def norm(grads: Sequence[tf.Tensor | None]) -> float:
        grads = [g for g in grads if g is not None]
        return 0.0 if not grads else float(tf.linalg.global_norm(grads).numpy())

    return {"data": norm(grad_data), "phys": norm(grad_phys), "ic": norm(grad_ic)}


# ============================================================
# 8. SAVE / LOAD MODEL
# ============================================================

def _save_model(model: keras.Model, path: str | os.PathLike[str]) -> str:
    path_obj = Path(path)
    if path_obj.suffix != ".keras":
        raise ValueError("Full-model checkpoints must use the .keras extension.")
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path_obj))
    return str(path_obj.resolve())


def _extract_scale_metadata(P: Mapping[str, Any]) -> dict[str, float] | None:
    """Get scaling metadata from either prepare_training_data or build_scaled_problem output."""
    scales = P.get("scales")
    if isinstance(scales, Mapping):
        try:
            return {key: float(scales[key]) for key in ("T0", "M0", "X0")}
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return {key: float(P[key]) for key in ("T0", "M0", "X0")}
    except (KeyError, TypeError, ValueError):
        return None


def _save_scale_metadata(scales: Mapping[str, float] | None, model_path: str | os.PathLike[str]) -> str | None:
    """Save the nondimensionalization scales next to a model checkpoint."""
    if scales is None:
        return None
    path_obj = Path(model_path)
    sidecar = path_obj.with_name(f"{path_obj.stem}_scales.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as file:
        json.dump({key: float(scales[key]) for key in ("T0", "M0", "X0")}, file, indent=2)
    return str(sidecar.resolve())


def load_trained_model(model_path: str | os.PathLike[str]) -> FourierPINN:
    """Load a saved FourierPINN .keras model in a separate prediction script."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")
    model = keras.models.load_model(
        str(path),
        custom_objects={"FourierPINN": FourierPINN},
        compile=False,
    )
    if not isinstance(model, FourierPINN):
        raise TypeError(f"The saved model at {path} is not a FourierPINN.")

    # Physical-unit predictions require the same scales used in training.
    scale_path = path.with_name(f"{path.stem}_scales.json")
    if scale_path.is_file():
        with scale_path.open("r", encoding="utf-8") as file:
            model._pinn_scales = json.load(file)
    return model


def check_save_load(
    model: FourierPINN,
    P: Mapping[str, Any],
    path: str | os.PathLike[str] = "tmp_roundtrip.keras",
) -> float:
    """Save/reload a model and report max prediction difference on 200 times.

    Run this once in the experiment environment before relying on checkpoints
    for inverse training. The requested checkpoint and scale sidecar are kept.
    """
    tau_np = _scaled_numpy(P, "tau")[:200]
    if len(tau_np) == 0:
        raise ValueError("P must contain at least one scaled time sample.")
    tau = tf.convert_to_tensor(tau_np, dtype=_FLOAT_DTYPE)
    before = model(tau).numpy()
    _save_model(model, path)
    _save_scale_metadata(_extract_scale_metadata(P), path)
    reloaded = load_trained_model(path)
    after = reloaded(tau).numpy()
    max_diff = float(np.max(np.abs(before - after)))
    print(f"Save/load round-trip max diff: {max_diff:.6e}")
    return max_diff


# ============================================================
# 9. TRAIN PINN: DATA WARM-UP + PHYSICS WEIGHT RAMP
# ============================================================

def train_pinn(
    model: FourierPINN,
    P: Mapping[str, Any],
    idx_data: Any | None = None,
    data_steps: int = 10_000,
    physics_steps_per_stage: int = 4_000,
    learning_rate: float = 1e-3,
    lambda_data: float = 1.0,
    lambda_ic: float = 1.0,
    physics_weights: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    best_physics_weight: float = 1e-2,
    n_col: int = 1024,
    log_every: int = 500,
    grad_every: int = 1000,
    save_dir: str | os.PathLike[str] = "models",
    model_name: str = "healthy_fourier_pinn",
    save_all_stages: bool = True,
    gradient_diagnostics: bool = True,
    test_a_tol: float = 1e-2,
    high_phys_lr: float = 3e-4,
    high_phys_threshold: float = 1e-2,
    decay_lr_within_stage: bool = False,
    stage_lr_decay_factor: float = 0.05,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Train a forward FourierPINN and save reusable checkpoints.

    Schedule: data+IC warm-up, then a physics-weight ramp. The model
    architecture defaults (w_max=8, n_freq=56, width=128) are unchanged so the
    previously measured baseline remains comparable. Set the random seed in
    the experiment script BEFORE build_model() to seed initial model weights;
    ``seed`` here is recorded in the results and controls gradient-diagnostic
    point selection, but deliberately does not reset weights after model build.
    If ``decay_lr_within_stage`` is True, the LR decays exponentially from each
    stage's configured start rate to ``stage_lr_decay_factor`` times that rate.
    """
    if data_steps < 1 or physics_steps_per_stage < 1:
        raise ValueError("data_steps and physics_steps_per_stage must be positive.")
    if n_col < 1 or log_every < 1 or grad_every < 1:
        raise ValueError("n_col, log_every, and grad_every must be positive.")
    if learning_rate <= 0 or high_phys_lr <= 0:
        raise ValueError("Learning rates must be positive.")
    if test_a_tol <= 0 or high_phys_threshold < 0:
        raise ValueError("test_a_tol must be positive and high_phys_threshold non-negative.")
    if not 0.0 < stage_lr_decay_factor <= 1.0:
        raise ValueError("stage_lr_decay_factor must be in (0, 1].")
    physics_weights = tuple(float(value) for value in physics_weights)
    if not physics_weights:
        raise ValueError("physics_weights cannot be empty.")
    if not any(np.isclose(value, best_physics_weight) for value in physics_weights):
        raise ValueError("best_physics_weight must be present in physics_weights.")

    P_t = _scaled_tensor_dict(P)
    tau_np = P_t["tau"].numpy()
    if idx_data is None:
        # Existing baseline: every fourth time point, all DOFs.
        idx_data = np.arange(0, len(tau_np), 4, dtype=np.int32)
    idx_data = tf.convert_to_tensor(idx_data, dtype=tf.int32)
    if int(tf.size(idx_data).numpy()) == 0:
        raise ValueError("idx_data must contain at least one observation index.")
    if int(tf.reduce_min(idx_data).numpy()) < 0 or int(tf.reduce_max(idx_data).numpy()) >= len(tau_np):
        raise ValueError("idx_data contains an out-of-range time index.")

    # The model should already have been built with a seeded initializer in
    # the experiment script; this call ensures variables exist for training.
    model(tf.convert_to_tensor(tau_np[:1], dtype=_FLOAT_DTYPE))

    weights = {
        "data": tf.Variable(lambda_data, dtype=_FLOAT_DTYPE, trainable=False),
        "phys": tf.Variable(0.0, dtype=_FLOAT_DTYPE, trainable=False),
        "ic": tf.Variable(lambda_ic, dtype=_FLOAT_DTYPE, trainable=False),
    }
    tau_window = tf.Variable(float(tau_np[-1, 0]), dtype=_FLOAT_DTYPE, trainable=False)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    scale_info = _extract_scale_metadata(P)
    if scale_info is not None:
        model._pinn_scales = scale_info.copy()

    base_dir = Path(save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    selected_path = base_dir / f"{model_name}.keras"
    warmup_path = base_dir / f"{model_name}_data_warmup.keras"
    final_path = base_dir / f"{model_name}_final.keras"

    history: dict[str, list[Any]] = {
        "stage": [], "epoch": [], "learning_rate": [],
        "total_loss": [], "data_loss": [], "physics_loss": [],
        "ic_loss": [], "metrics": [], "gradient_norms": [],
    }
    stage_metrics: dict[str, dict[str, float]] = {}
    stage_model_paths: dict[str, str] = {}

    def run_stage(stage_name: str, steps: int, data_only: bool) -> dict[str, float]:
        lr_start = float(optimizer.learning_rate.numpy())
        print("\n" + "=" * 78)
        print(stage_name)
        print(f"Steps: {steps}")
        print(f"Starting learning rate: {lr_start:.2e}")
        if decay_lr_within_stage:
            lr_end = lr_start * stage_lr_decay_factor
            print(
                f"Within-stage LR decay: {stage_lr_decay_factor:g}x "
                f"({lr_start:.2e} -> {lr_end:.2e})"
            )
        print(f"Physics weight: {float(weights['phys'].numpy()):.2e}")
        print("=" * 78)

        for epoch in range(1, steps + 1):
            if decay_lr_within_stage:
                current_stage_lr = lr_start * (
                    stage_lr_decay_factor ** (epoch / steps)
                )
                optimizer.learning_rate.assign(current_stage_lr)

            total, Ld, Lp, Li = train_step(
                model=model,
                optimizer=optimizer,
                P=P_t,
                idx_data=idx_data,
                w=weights,
                n_col=n_col,
                tau_window=tau_window,
                data_only=data_only,
            )

            # Evaluate only at reporting points; the final epoch is always logged.
            if epoch % log_every == 0 or epoch == steps:
                loss_values = {
                    "total": float(total.numpy()),
                    "data": float(Ld.numpy()),
                    "phys": float(Lp.numpy()),
                    "ic": float(Li.numpy()),
                }
                metrics = evaluate_model(model, P_t)
                history["stage"].append(stage_name)
                history["epoch"].append(epoch)
                history["learning_rate"].append(float(optimizer.learning_rate.numpy()))
                history["total_loss"].append(loss_values["total"])
                history["data_loss"].append(loss_values["data"])
                history["physics_loss"].append(loss_values["phys"])
                history["ic_loss"].append(loss_values["ic"])
                history["metrics"].append(metrics)
                print(
                    f"[{stage_name}] ep={epoch:5d}/{steps} | "
                    f"lr={float(optimizer.learning_rate.numpy()):.2e} | "
                    f"total={loss_values['total']:.2e} | "
                    f"data={loss_values['data']:.2e} | "
                    f"phys={loss_values['phys']:.2e} | "
                    f"ic={loss_values['ic']:.2e} | "
                    f"relL2 x={metrics['x']:.2e} | "
                    f"v={metrics['v']:.2e} | a={metrics['a']:.2e}"
                )

            if gradient_diagnostics and epoch % grad_every == 0:
                gn = get_gradient_norms(
                    model=model,
                    P=P_t,
                    idx_data=idx_data,
                    n_col=min(256, n_col),
                    tau_window=float(tau_window.numpy()),
                    seed=seed,
                )
                history["gradient_norms"].append({"stage": stage_name, "epoch": epoch, **gn})
                ratio = gn["phys"] / max(gn["data"], 1e-30)
                print(
                    f"  Gradient norms | data={gn['data']:.2e} | "
                    f"phys={gn['phys']:.2e} | ic={gn['ic']:.2e} | "
                    f"phys/data={ratio:.2e}"
                )

        # Every positive-length stage logs at its final epoch.
        return history["metrics"][-1]

    # Test A: data + initial conditions only.
    weights["phys"].assign(0.0)
    optimizer.learning_rate.assign(learning_rate)
    metrics_A = run_stage("WARM-UP: DATA + IC", data_steps, data_only=True)
    test_A_passed = metrics_A["x"] < test_a_tol
    print("\nWARM-UP FINAL RESULTS")
    print(f"relL2 x={metrics_A['x']:.4e} | v={metrics_A['v']:.4e} | a={metrics_A['a']:.4e}")

    # Save the warm-up checkpoint regardless, so it is available for inspection.
    _save_model(model, warmup_path)
    _save_scale_metadata(scale_info, warmup_path)
    model_paths: dict[str, str] = {"data_warmup": str(warmup_path.resolve())}

    if not test_A_passed:
        print(
            "TEST B skipped because displacement relative L2 error "
            "did not fall below the tolerance."
        )
        return {
            "model": model,
            "history": history,
            "metrics_A": metrics_A,
            "test_A_passed": False,
            "stage_metrics": stage_metrics,
            "model_paths": model_paths,
            "selected_model_path": None,
            "final_model_path": None,
            "seed": seed,
            "decay_lr_within_stage": bool(decay_lr_within_stage),
            "stage_lr_decay_factor": float(stage_lr_decay_factor),
        }

    print("\nStarting PHYSICS RAMP from the warm-up weights.")
    for stage_weight in physics_weights:
        weights["phys"].assign(stage_weight)
        current_lr = high_phys_lr if stage_weight >= high_phys_threshold else learning_rate
        optimizer.learning_rate.assign(current_lr)
        stage_name = f"PHYSICS RAMP: w={stage_weight:.0e}"
        metrics = run_stage(stage_name, physics_steps_per_stage, data_only=False)
        stage_key = f"phys_{stage_weight:.0e}"
        stage_metrics[stage_key] = metrics
        print(
            f"After physics weight {stage_weight:.0e}: "
            f"relL2 x={metrics['x']:.4e} | "
            f"v={metrics['v']:.4e} | a={metrics['a']:.4e}"
        )

        if save_all_stages:
            stage_path = base_dir / f"{model_name}_{stage_key}.keras"
            stage_model_paths[stage_key] = _save_model(model, stage_path)
            _save_scale_metadata(scale_info, stage_path)

        if np.isclose(stage_weight, best_physics_weight):
            model_paths["selected"] = _save_model(model, selected_path)
            _save_scale_metadata(scale_info, selected_path)

    model_paths["final"] = _save_model(model, final_path)
    _save_scale_metadata(scale_info, final_path)
    model_paths.update(stage_model_paths)
    model_paths["data_warmup"] = str(warmup_path.resolve())

    print("\nTRAINING COMPLETE")
    print(
        f"Selected checkpoint (physics weight {best_physics_weight:g}): "
        f"{model_paths.get('selected', str(selected_path.resolve()))}"
    )
    print(f"Final-stage checkpoint: {model_paths['final']}")

    return {
        "model": model,
        "history": history,
        "metrics_A": metrics_A,
        "test_A_passed": True,
        "stage_metrics": stage_metrics,
        "model_paths": model_paths,
        "selected_model_path": model_paths.get("selected", str(selected_path.resolve())),
        "final_model_path": model_paths["final"],
        "seed": seed,
        "decay_lr_within_stage": bool(decay_lr_within_stage),
        "stage_lr_decay_factor": float(stage_lr_decay_factor),
    }


# ============================================================
# 10. PREDICT RESPONSE
# ============================================================

def predict_response(
    model: keras.Model | str | os.PathLike[str],
    t: Any,
    scales: Mapping[str, float] | None = None,
    time_is_physical: bool = False,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict displacement, velocity, and acceleration.

    Args:
        model: A loaded FourierPINN or path to a saved .keras model.
        t: Dimensionless time tau by default. If time_is_physical=True,
           t must be physical time in seconds and scales must be provided.
        scales: Optional {'T0','M0','X0'} dictionary from healthy preparation.
               If supplied, outputs are converted back to physical units.
        time_is_physical: Convert physical time t to tau=t/T0 before prediction.
        batch_size: Batch size for autodiff to reduce memory use.

    Returns:
        Tuple (displacement, velocity, acceleration). Outputs are dimensionless
        when scales=None and in physical units when scales is provided.
    """
    if isinstance(model, (str, os.PathLike)):
        model = load_trained_model(model)

    if scales is None:
        scales = getattr(model, "_pinn_scales", None)
    if scales is None:
        if time_is_physical:
            raise ValueError("Pass saved scales when time_is_physical=True.")
        warnings.warn(
            "No scales found: returning dimensionless outputs.",
            stacklevel=2,
        )

    time_array = np.asarray(t.numpy() if tf.is_tensor(t) else t, dtype=np.float64)
    if time_array.ndim == 1:
        time_array = time_array.reshape(-1, 1)
    if time_array.ndim != 2 or time_array.shape[1] != 1:
        raise ValueError("t must have shape (n_time,) or (n_time, 1).")

    if time_is_physical:
        time_array = time_array / float(scales["T0"])

    x_pred, v_pred, a_pred = derivatives_in_batches(model, time_array, batch_size=batch_size)

    if scales is not None:
        T0 = float(scales["T0"])
        X0 = float(scales["X0"])
        x_pred = x_pred * X0
        v_pred = v_pred * X0 / T0
        a_pred = a_pred * X0 / (T0**2)

    return x_pred, v_pred, a_pred


