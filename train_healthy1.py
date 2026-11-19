"""Separate experiment: healthy FourierPINN with within-stage LR decay.

This file calls the reusable functions in pinn_model.py. It contains no model
architecture or loss-function definitions; those live in pinn_model.py.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from datasets import prepare_healthy_pinn_data
from models1 import (
    build_model,
    calculate_physics_residual,
    check_save_load,
    evaluate_model,
    load_trained_model,
    predict_response,
    prepare_training_data,
    train_pinn,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 0
MODEL_DIR = Path("models1/weights")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Keep the established Test A / Test B baseline.
W_MAX = 8.0
N_FREQ = 56
WIDTH = 128
DEPTH = 4

DATA_STEPS = 10_000
PHYSICS_STEPS_PER_STAGE = 4_000
PHYSICS_WEIGHTS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BEST_PHYSICS_WEIGHT = 1e-2
LEARNING_RATE = 1e-3
HIGH_PHYS_LR = 3e-4
HIGH_PHYS_THRESHOLD = 1e-2
N_COL = 1024
LOG_EVERY = 500
GRAD_EVERY = 1000
TEST_A_TOL = 1e-2
STAGE_LR_DECAY_FACTOR = 0.05
DECAY_LR_WITHIN_STAGE = True

# Turn off to skip extra physics diagnostics and save/load round-trip test.
RUN_DIAGNOSTICS = True
RUN_SAVE_LOAD_CHECK = True

MODEL_NAME = "healthy_fourier_pinn_lrdecay"


# ============================================================
# OPTIONAL DIAGNOSTICS
# ============================================================

def print_data_summary(case, P):
    """Print compact shapes/ranges without building pandas DataFrames."""
    print("\n========== DATA SUMMARY ==========")
    for key in ("tau", "x", "v", "a", "F"):
        arr = P[key].numpy()
        print(
            f"{key:>5}: shape={arr.shape}, "
            f"min={np.min(arr):.6e}, max={np.max(arr):.6e}, "
            f"RMS={np.sqrt(np.mean(arr**2)):.6e}"
        )

    print("M shape:", P["M"].shape)
    print("C shape:", P["C"].shape)
    print("K shape:", P["K"].shape)
    print("Scaling values:", P["scales"])


def true_data_physics_check(P):
    """Check the scaled simulation states against the scaled equation of motion."""
    x = P["x"].numpy()
    v = P["v"].numpy()
    a = P["a"].numpy()
    M = P["M"].numpy()
    C = P["C"].numpy()
    K = P["K"].numpy()
    F = P["F"].numpy()

    residual = a @ M.T + v @ C.T + x @ K.T - F
    print("\n========== TRUE-DATA PHYSICS CHECK (SCALED) ==========")
    print(f"MSE : {np.mean(residual**2):.8e}")
    print(f"RMSE: {np.sqrt(np.mean(residual**2)):.8e}")
    print(f"Max : {np.max(np.abs(residual)):.8e}")


def model_physics_check(model, P, batch_size=512):
    """Compute PINN physics-residual metrics on a limited time prefix."""
    tau = P["tau"][:512]
    force = P["F"][:512]
    residual_parts = []

    for start in range(0, int(tau.shape[0]), batch_size):
        stop = start + batch_size
        _, _, _, residual = calculate_physics_residual(
            model=model,
            t=tau[start:stop],
            M=P["M"],
            C=P["C"],
            K=P["K"],
            force_batch=force[start:stop],
        )
        residual_parts.append(residual.numpy())

    residual = np.concatenate(residual_parts, axis=0)
    print("\n========== PINN PHYSICS CHECK (SCALED) ==========")
    print(f"MSE : {np.mean(residual**2):.8e}")
    print(f"RMSE: {np.sqrt(np.mean(residual**2)):.8e}")
    print(f"Max : {np.max(np.abs(residual)):.8e}")


# ============================================================
# 1. PREPARE HEALTHY CASE AND SCALED DATA
# ============================================================

# Seed before model construction so the initial weights are reproducible.
tf.keras.utils.set_random_seed(SEED)

case_healthy = prepare_healthy_pinn_data()
print("Available physical-case keys:", list(case_healthy.keys()))

P = prepare_training_data(
    case=case_healthy,
    K_ref=np.asarray(case_healthy["stiffness_matrix"], dtype=np.float64),
    scales=None,
    dtype=tf.float64,
    expose_K=True,
)

if RUN_DIAGNOSTICS:
    print_data_summary(case_healthy, P)
    true_data_physics_check(P)


# ============================================================
# 2. BUILD THE FOURIER PINN
# ============================================================

model = build_model(
    P=P,
    w_max=W_MAX,
    n_freq=N_FREQ,
    width=WIDTH,
    depth=DEPTH,
)

print("\n========== MODEL CONFIGURATION ==========")
print(f"DOF: {int(P['x'].shape[1])}")
print(f"Fourier max frequency: {W_MAX}")
print(f"Number of Fourier frequencies: {N_FREQ}")
print(f"Width: {WIDTH}; depth: {DEPTH}")
print(f"Tensor dtype: {P['x'].dtype.name}")


# ============================================================
# 3. TRAIN: DATA WARM-UP + PHYSICS-WEIGHT RAMP
# ============================================================

results = train_pinn(
    model=model,
    P=P,
    data_steps=DATA_STEPS,
    physics_steps_per_stage=PHYSICS_STEPS_PER_STAGE,
    learning_rate=LEARNING_RATE,
    lambda_data=1.0,
    lambda_ic=1.0,
    physics_weights=PHYSICS_WEIGHTS,
    best_physics_weight=BEST_PHYSICS_WEIGHT,
    n_col=N_COL,
    log_every=LOG_EVERY,
    grad_every=GRAD_EVERY,
    save_dir=MODEL_DIR,
    model_name=MODEL_NAME,
    save_all_stages=True,
    gradient_diagnostics=True,
    test_a_tol=TEST_A_TOL,
    high_phys_lr=HIGH_PHYS_LR,
    high_phys_threshold=HIGH_PHYS_THRESHOLD,
    decay_lr_within_stage=DECAY_LR_WITHIN_STAGE,
    stage_lr_decay_factor=STAGE_LR_DECAY_FACTOR,
    seed=SEED,
)

# The reusable trainer saves full .keras checkpoints and scale sidecars.
# This driver saves the returned history and explicit run configuration.
with (MODEL_DIR / f"{MODEL_NAME}_history.json").open("w", encoding="utf-8") as file:
    json.dump(results["history"], file, indent=2, allow_nan=False)

run_config = {
    "seed": SEED,
    "architecture": {
        "type": "FourierPINN",
        "w_max": W_MAX,
        "n_freq": N_FREQ,
        "width": WIDTH,
        "depth": DEPTH,
        "dof": int(P["x"].shape[1]),
    },
    "training": {
        "experiment_variant": "within_stage_exponential_lr_decay",
        "data_steps": DATA_STEPS,
        "physics_steps_per_stage": PHYSICS_STEPS_PER_STAGE,
        "physics_weights": list(PHYSICS_WEIGHTS),
        "best_physics_weight": BEST_PHYSICS_WEIGHT,
        "learning_rate": LEARNING_RATE,
        "high_phys_lr": HIGH_PHYS_LR,
        "high_phys_threshold": HIGH_PHYS_THRESHOLD,
        "decay_lr_within_stage": DECAY_LR_WITHIN_STAGE,
        "stage_lr_decay_factor": STAGE_LR_DECAY_FACTOR,
        "n_col": N_COL,
        "log_every": LOG_EVERY,
        "grad_every": GRAD_EVERY,
        "test_a_tol": TEST_A_TOL,
        "test_A_passed": bool(results["test_A_passed"]),
    },
    "scales": P["scales"],
    "model_paths": results["model_paths"],
    "selected_model_path": results["selected_model_path"],
    "final_model_path": results["final_model_path"],
}
with (MODEL_DIR / f"{MODEL_NAME}_run_config.json").open("w", encoding="utf-8") as file:
    json.dump(run_config, file, indent=2, allow_nan=False)

print("\n========== TRAINING RESULT ==========")
print("Test A passed:", results["test_A_passed"])
print("Selected checkpoint:", results["selected_model_path"])
print("Final checkpoint:", results["final_model_path"])
print("History saved to:", MODEL_DIR / f"{MODEL_NAME}_history.json")


# ============================================================
# 4. LOAD CHECKPOINT AND PREDICT
# ============================================================

# Prefer the selected checkpoint (physics weight 1e-2). If warm-up failed,
# use the warm-up checkpoint only for diagnostic predictions; it is NOT marked
# as the selected/final model by train_pinn().
if results["selected_model_path"] is not None:
    prediction_model_path = results["selected_model_path"]
    prediction_checkpoint_type = "selected physics-ramp checkpoint"
else:
    prediction_model_path = results["model_paths"]["data_warmup"]
    prediction_checkpoint_type = "warm-up fallback (Test B was skipped)"

print("\nPrediction checkpoint:", prediction_checkpoint_type)
print("Path:", prediction_model_path)

if RUN_SAVE_LOAD_CHECK:
    loaded_for_check = load_trained_model(prediction_model_path)
    check_save_load(
        model=loaded_for_check,
        P=P,
        path=MODEL_DIR / f"{MODEL_NAME}_roundtrip_check.keras",
    )

# Pass physical seconds; saved scale metadata converts them to dimensionless tau
# for the model and converts predicted x/v/a back to physical units.
time_physical = np.asarray(case_healthy["time"], dtype=np.float64)
x_pred, v_pred, a_pred = predict_response(
    model=prediction_model_path,
    t=time_physical,
    time_is_physical=True,
    batch_size=512,
)

print("\n========== PREDICTION SUMMARY (PHYSICAL UNITS) ==========")
for name, arr in (("x_pred", x_pred), ("v_pred", v_pred), ("a_pred", a_pred)):
    print(
        f"{name}: shape={arr.shape}, min={np.min(arr):.6e}, "
        f"max={np.max(arr):.6e}, RMS={np.sqrt(np.mean(arr**2)):.6e}"
    )

prediction_path = MODEL_DIR / f"{MODEL_NAME}_predictions.npz"
np.savez_compressed(
    prediction_path,
    time=time_physical,
    displacement=x_pred,
    velocity=v_pred,
    acceleration=a_pred,
)
print("Predictions saved to:", prediction_path)

# Relative errors are calculated in scaled space against the same healthy case.
loaded_model = load_trained_model(prediction_model_path)
metrics = evaluate_model(loaded_model, P)
print("\n========== FULL-FIELD RELATIVE L2 ERRORS (SCALED) ==========")
print(f"Displacement: {metrics['x']:.6e}")
print(f"Velocity:     {metrics['v']:.6e}")
print(f"Acceleration: {metrics['a']:.6e}")

if RUN_DIAGNOSTICS:
    model_physics_check(loaded_model, P)

print("\n========== HEALTHY PINN EXPERIMENT COMPLETE ==========")
