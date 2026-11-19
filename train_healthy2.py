"""Reproducible healthy FourierPINN run / frequency-range comparison.

Place this file in the PINN project root, beside datasets.py and pinn_model.py.
Run controlled Fourier-frequency experiments with the same seed.

Examples:
    python run_healthy_seeded.py --w-max 8 --n-freq 56 --seed 42
    python run_healthy_seeded.py --w-max 5 --n-freq 32 --seed 42

The two runs write to separate directories and do not overwrite the existing
models/weights or models1/weights checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from datasets import prepare_healthy_pinn_data
from models1 import (
    build_model,
    check_save_load,
    evaluate_model,
    load_trained_model,
    prepare_training_data,
    train_pinn,
)


def fmt_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--w-max", type=float, required=True, help="Maximum Fourier frequency; must be > 0.25")
    parser.add_argument("--n-freq", type=int, default=56)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("models1/experiments/optional_checks"))
    args = parser.parse_args()

    # Match all non-frequency training settings to the current LR-decay run.
    W_MAX = float(args.w_max)
    N_FREQ = int(args.n_freq)
    if W_MAX <= 0.25:
        parser.error("--w-max must be greater than 0.25, the minimum Fourier frequency")
    if N_FREQ < 2:
        parser.error("--n-freq must be at least 2")
    WIDTH, DEPTH = 128, 4
    DATA_STEPS = 10_000
    PHYSICS_STEPS_PER_STAGE = 4_000
    PHYSICS_WEIGHTS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    BEST_PHYSICS_WEIGHT = 1e-2
    LEARNING_RATE = 1e-3
    HIGH_PHYS_LR = 3e-4
    HIGH_PHYS_THRESHOLD = 1e-2
    N_COL = 1024
    LOG_EVERY, GRAD_EVERY = 500, 1000
    TEST_A_TOL = 1e-2
    DECAY_FACTOR = 0.05

    run_name = f"seed_{args.seed}_wmax_{fmt_number(W_MAX)}_nfreq_{N_FREQ}"
    model_dir = args.output_root / run_name / "weights"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = "healthy_fourier_pinn_lrdecay"

    # Each (seed, w_max, n_freq) combination gets its own output directory.
    # Do not reuse an existing run directory if you need to preserve its artifacts.

    # Reset before model construction so both frequency runs start from the
    # same seeded initialization stream and use the same random seed.
    tf.keras.utils.set_random_seed(args.seed)
    case_h = prepare_healthy_pinn_data()
    P = prepare_training_data(
        case=case_h,
        K_ref=np.asarray(case_h["stiffness_matrix"], dtype=np.float64),
        scales=None,
        dtype=tf.float64,
        expose_K=True,
    )

    model = build_model(P=P, w_max=W_MAX, n_freq=N_FREQ, width=WIDTH, depth=DEPTH)
    print("\n========== REPRODUCIBLE HEALTHY PINN RUN ==========")
    print(f"seed={args.seed} | w_max={W_MAX:g} | n_freq={N_FREQ} | width={WIDTH} | depth={DEPTH}")
    print(f"Frequency minimum=0.25; frequency spacing={(W_MAX - 0.25) / max(N_FREQ - 1, 1):.8g}")
    print("scales:", P["scales"])
    print("output directory:", model_dir)

    result = train_pinn(
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
        save_dir=model_dir,
        model_name=model_name,
        save_all_stages=True,
        gradient_diagnostics=True,
        test_a_tol=TEST_A_TOL,
        high_phys_lr=HIGH_PHYS_LR,
        high_phys_threshold=HIGH_PHYS_THRESHOLD,
        decay_lr_within_stage=True,
        stage_lr_decay_factor=DECAY_FACTOR,
        seed=args.seed,
    )

    history_path = model_dir / f"{model_name}_history.json"
    history_path.write_text(json.dumps(result["history"], indent=2, allow_nan=False), encoding="utf-8")

    selected_path = result.get("selected_model_path")
    metrics = None
    roundtrip_diff = None
    if selected_path:
        loaded = load_trained_model(selected_path)
        metrics = evaluate_model(loaded, P)
        roundtrip_diff = check_save_load(
            model=loaded,
            P=P,
            path=model_dir / f"{model_name}_roundtrip_check.keras",
        )

    config = {
        "seed": args.seed,
        "architecture": {
            "type": "FourierPINN",
            "dof": int(P["x"].shape[1]),
            "w_max": W_MAX,
            "n_freq": N_FREQ,
            "frequency_min": 0.25,
            "frequency_spacing": (W_MAX - 0.25) / max(N_FREQ - 1, 1),
            "width": WIDTH,
            "depth": DEPTH,
            "dtype": P["x"].dtype.name,
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
            "decay_lr_within_stage": True,
            "stage_lr_decay_factor": DECAY_FACTOR,
            "n_col": N_COL,
            "log_every": LOG_EVERY,
            "grad_every": GRAD_EVERY,
            "test_a_tolerance": TEST_A_TOL,
        },
        "scales": P["scales"],
        "test_A_passed": bool(result["test_A_passed"]),
        "stage_metrics": result.get("stage_metrics", {}),
        "selected_checkpoint": selected_path,
        "final_checkpoint": result.get("final_model_path"),
        "selected_checkpoint_metrics": metrics,
        "save_load_max_abs_difference": roundtrip_diff,
        "model_paths": result.get("model_paths", {}),
    }
    config_path = model_dir / f"{model_name}_run_config.json"
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf-8")
    print("\nSaved run config:", config_path)
    print("Saved history:", history_path)
    print("Selected checkpoint:", selected_path)
    print("Selected metrics:", metrics)


if __name__ == "__main__":
    main()


#\(w_{\max}=8,\quad n_{\text{freq}}=56\)

'''
(.venv) manjeet_kumar@DESKTOP-NV4RH6T:/mnt/k/Projects/PINNS$ python train_healthy2.py \
  --w-max 8 \
  --n-freq 56 \
  --seed 42 \
  --output-root models1/experiments/optional_checks
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792155016.275863    5538 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792155018.249888    5538 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792155048.757626    5538 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Traceback (most recent call last):
  File "/mnt/k/Projects/PINNS/train_healthy2.py", line 23, in <module>
    from pinn_model import (
ModuleNotFoundError: No module named 'pinn_model'
(.venv) manjeet_kumar@DESKTOP-NV4RH6T:/mnt/k/Projects/PINNS$ python train_healthy2.py   --w-max 8   --n-freq 56   --seed 42   --output-root models1/experiments/optional_checks
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792155538.697980    5851 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792155540.028536    5851 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792155570.811006    5851 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792155595.760562    5851 gpu_device.cc:2043] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3537 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4050 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9

========== REPRODUCIBLE HEALTHY PINN RUN ==========
seed=42 | w_max=8 | n_freq=56 | width=128 | depth=4
Frequency minimum=0.25; frequency spacing=0.14090909
scales: {'T0': 0.21157993442837367, 'M0': 1.0, 'X0': 0.3941715948658804}
output directory: models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights

==============================================================================
WARM-UP: DATA + IC
Steps: 10000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 0.00e+00
==============================================================================
[WARM-UP: DATA + IC] ep=  500/10000 | lr=8.61e-04 | total=3.73e-04 | data=3.73e-04 | phys=0.00e+00 | ic=9.98e-09 | relL2 x=1.93e-02 | v=3.59e-01 | a=8.00e+00
[WARM-UP: DATA + IC] ep= 1000/10000 | lr=7.41e-04 | total=1.18e-04 | data=1.18e-04 | phys=0.00e+00 | ic=2.23e-09 | relL2 x=1.09e-02 | v=2.41e-01 | a=5.99e+00
  Gradient norms | data=5.62e-04 | phys=8.91e+02 | ic=3.72e-04 | phys/data=1.59e+06
[WARM-UP: DATA + IC] ep= 1500/10000 | lr=6.38e-04 | total=6.30e-05 | data=6.30e-05 | phys=0.00e+00 | ic=1.16e-09 | relL2 x=7.96e-03 | v=1.93e-01 | a=5.09e+00
[WARM-UP: DATA + IC] ep= 2000/10000 | lr=5.49e-04 | total=3.93e-05 | data=3.93e-05 | phys=0.00e+00 | ic=6.00e-10 | relL2 x=6.30e-03 | v=1.63e-01 | a=4.49e+00
  Gradient norms | data=2.25e-04 | phys=6.05e+02 | ic=1.72e-04 | phys/data=2.69e+06
[WARM-UP: DATA + IC] ep= 2500/10000 | lr=4.73e-04 | total=2.79e-05 | data=2.77e-05 | phys=0.00e+00 | ic=1.45e-07 | relL2 x=5.29e-03 | v=1.43e-01 | a=4.06e+00
[WARM-UP: DATA + IC] ep= 3000/10000 | lr=4.07e-04 | total=4.90e-05 | data=2.22e-05 | phys=0.00e+00 | ic=2.67e-05 | relL2 x=4.63e-03 | v=1.28e-01 | a=3.72e+00
  Gradient norms | data=7.60e-04 | phys=4.78e+02 | ic=1.86e-02 | phys/data=6.29e+05
[WARM-UP: DATA + IC] ep= 3500/10000 | lr=3.50e-04 | total=2.09e-05 | data=1.68e-05 | phys=0.00e+00 | ic=4.15e-06 | relL2 x=4.13e-03 | v=1.16e-01 | a=3.46e+00
[WARM-UP: DATA + IC] ep= 4000/10000 | lr=3.02e-04 | total=1.34e-05 | data=1.34e-05 | phys=0.00e+00 | ic=9.76e-09 | relL2 x=3.68e-03 | v=1.07e-01 | a=3.23e+00
  Gradient norms | data=1.21e-04 | phys=4.01e+02 | ic=6.41e-04 | phys/data=3.30e+06
[WARM-UP: DATA + IC] ep= 4500/10000 | lr=2.60e-04 | total=1.58e-05 | data=1.13e-05 | phys=0.00e+00 | ic=4.46e-06 | relL2 x=3.38e-03 | v=9.96e-02 | a=3.05e+00
[WARM-UP: DATA + IC] ep= 5000/10000 | lr=2.24e-04 | total=9.43e-06 | data=9.43e-06 | phys=0.00e+00 | ic=3.78e-10 | relL2 x=3.09e-03 | v=9.32e-02 | a=2.90e+00
  Gradient norms | data=1.06e-04 | phys=3.53e+02 | ic=6.81e-05 | phys/data=3.34e+06
[WARM-UP: DATA + IC] ep= 5500/10000 | lr=1.93e-04 | total=8.35e-06 | data=8.12e-06 | phys=0.00e+00 | ic=2.29e-07 | relL2 x=2.87e-03 | v=8.77e-02 | a=2.76e+00
[WARM-UP: DATA + IC] ep= 6000/10000 | lr=1.66e-04 | total=2.71e-05 | data=7.37e-06 | phys=0.00e+00 | ic=1.98e-05 | relL2 x=2.74e-03 | v=8.31e-02 | a=2.64e+00
  Gradient norms | data=1.08e-03 | phys=3.22e+02 | ic=4.45e-02 | phys/data=2.99e+05
[WARM-UP: DATA + IC] ep= 6500/10000 | lr=1.43e-04 | total=6.21e-06 | data=6.18e-06 | phys=0.00e+00 | ic=2.88e-08 | relL2 x=2.51e-03 | v=7.89e-02 | a=2.54e+00
[WARM-UP: DATA + IC] ep= 7000/10000 | lr=1.23e-04 | total=5.57e-06 | data=5.47e-06 | phys=0.00e+00 | ic=1.06e-07 | relL2 x=2.36e-03 | v=7.52e-02 | a=2.44e+00
  Gradient norms | data=9.76e-05 | phys=2.98e+02 | ic=1.93e-03 | phys/data=3.06e+06
[WARM-UP: DATA + IC] ep= 7500/10000 | lr=1.06e-04 | total=5.75e-06 | data=4.90e-06 | phys=0.00e+00 | ic=8.53e-07 | relL2 x=2.23e-03 | v=7.19e-02 | a=2.36e+00
[WARM-UP: DATA + IC] ep= 8000/10000 | lr=9.10e-05 | total=4.41e-06 | data=4.39e-06 | phys=0.00e+00 | ic=2.36e-08 | relL2 x=2.12e-03 | v=6.90e-02 | a=2.28e+00
  Gradient norms | data=9.79e-05 | phys=2.80e+02 | ic=9.97e-04 | phys/data=2.87e+06
[WARM-UP: DATA + IC] ep= 8500/10000 | lr=7.84e-05 | total=4.14e-06 | data=3.98e-06 | phys=0.00e+00 | ic=1.60e-07 | relL2 x=2.01e-03 | v=6.64e-02 | a=2.22e+00
[WARM-UP: DATA + IC] ep= 9000/10000 | lr=6.75e-05 | total=3.65e-06 | data=3.62e-06 | phys=0.00e+00 | ic=3.69e-08 | relL2 x=1.92e-03 | v=6.40e-02 | a=2.15e+00
  Gradient norms | data=9.53e-05 | phys=2.66e+02 | ic=1.41e-03 | phys/data=2.79e+06
[WARM-UP: DATA + IC] ep= 9500/10000 | lr=5.81e-05 | total=3.33e-06 | data=3.31e-06 | phys=0.00e+00 | ic=2.23e-08 | relL2 x=1.84e-03 | v=6.18e-02 | a=2.09e+00
[WARM-UP: DATA + IC] ep=10000/10000 | lr=5.00e-05 | total=3.05e-06 | data=3.04e-06 | phys=0.00e+00 | ic=4.02e-09 | relL2 x=1.76e-03 | v=5.98e-02 | a=2.04e+00
  Gradient norms | data=5.52e-05 | phys=2.53e+02 | ic=5.70e-04 | phys/data=4.58e+06

WARM-UP FINAL RESULTS
relL2 x=1.7629e-03 | v=5.9781e-02 | a=2.0388e+00

Starting PHYSICS RAMP from the warm-up weights.

==============================================================================
PHYSICS RAMP: w=1e-04
Steps: 4000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 1.00e-04
==============================================================================
[PHYSICS RAMP: w=1e-04] ep=  500/4000 | lr=6.88e-04 | total=9.67e-05 | data=1.10e-05 | phys=8.56e-01 | ic=1.14e-07 | relL2 x=3.33e-03 | v=4.56e-02 | a=8.15e-01
[PHYSICS RAMP: w=1e-04] ep= 1000/4000 | lr=4.73e-04 | total=4.69e-05 | data=4.00e-06 | phys=4.28e-01 | ic=4.85e-08 | relL2 x=2.00e-03 | v=2.86e-02 | a=5.49e-01
  Gradient norms | data=5.01e-04 | phys=3.20e+01 | ic=1.16e-03 | phys/data=6.39e+04
[PHYSICS RAMP: w=1e-04] ep= 1500/4000 | lr=3.25e-04 | total=3.15e-05 | data=2.16e-06 | phys=2.93e-01 | ic=1.60e-08 | relL2 x=1.47e-03 | v=2.16e-02 | a=4.37e-01
[PHYSICS RAMP: w=1e-04] ep= 2000/4000 | lr=2.24e-04 | total=2.31e-05 | data=1.38e-06 | phys=2.17e-01 | ic=2.18e-08 | relL2 x=1.17e-03 | v=1.76e-02 | a=3.73e-01
  Gradient norms | data=2.78e-04 | phys=2.48e+01 | ic=7.63e-04 | phys/data=8.91e+04
[PHYSICS RAMP: w=1e-04] ep= 2500/4000 | lr=1.54e-04 | total=1.88e-05 | data=9.82e-07 | phys=1.78e-01 | ic=1.21e-08 | relL2 x=9.95e-04 | v=1.51e-02 | a=3.33e-01
[PHYSICS RAMP: w=1e-04] ep= 3000/4000 | lr=1.06e-04 | total=1.68e-05 | data=7.58e-07 | phys=1.60e-01 | ic=1.49e-08 | relL2 x=8.70e-04 | v=1.34e-02 | a=3.05e-01
  Gradient norms | data=2.11e-04 | phys=2.05e+01 | ic=8.40e-04 | phys/data=9.70e+04
[PHYSICS RAMP: w=1e-04] ep= 3500/4000 | lr=7.27e-05 | total=1.46e-05 | data=6.11e-07 | phys=1.39e-01 | ic=5.12e-09 | relL2 x=7.81e-04 | v=1.22e-02 | a=2.85e-01
[PHYSICS RAMP: w=1e-04] ep= 4000/4000 | lr=5.00e-05 | total=1.34e-05 | data=5.12e-07 | phys=1.29e-01 | ic=9.61e-09 | relL2 x=7.16e-04 | v=1.12e-02 | a=2.70e-01
  Gradient norms | data=2.08e-04 | phys=2.00e+01 | ic=4.42e-04 | phys/data=9.59e+04
After physics weight 1e-04: relL2 x=7.1641e-04 | v=1.1235e-02 | a=2.7011e-01

==============================================================================
PHYSICS RAMP: w=1e-03
Steps: 4000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 1.00e-03
==============================================================================
[PHYSICS RAMP: w=1e-03] ep=  500/4000 | lr=6.88e-04 | total=1.03e-04 | data=8.19e-06 | phys=9.39e-02 | ic=5.05e-07 | relL2 x=2.85e-03 | v=2.87e-02 | a=3.71e-01
[PHYSICS RAMP: w=1e-03] ep= 1000/4000 | lr=4.73e-04 | total=4.45e-05 | data=3.02e-06 | phys=4.14e-02 | ic=9.33e-08 | relL2 x=1.74e-03 | v=1.93e-02 | a=2.59e-01
  Gradient norms | data=6.74e-04 | phys=1.40e+01 | ic=2.12e-03 | phys/data=2.08e+04
[PHYSICS RAMP: w=1e-03] ep= 1500/4000 | lr=3.25e-04 | total=3.08e-05 | data=1.72e-06 | phys=2.90e-02 | ic=8.77e-08 | relL2 x=1.31e-03 | v=1.53e-02 | a=2.09e-01
[PHYSICS RAMP: w=1e-03] ep= 2000/4000 | lr=2.24e-04 | total=2.37e-05 | data=1.17e-06 | phys=2.25e-02 | ic=1.32e-08 | relL2 x=1.08e-03 | v=1.30e-02 | a=1.80e-01
  Gradient norms | data=3.98e-04 | phys=8.32e+00 | ic=8.55e-04 | phys/data=2.09e+04
[PHYSICS RAMP: w=1e-03] ep= 2500/4000 | lr=1.54e-04 | total=1.89e-05 | data=8.83e-07 | phys=1.79e-02 | ic=6.41e-08 | relL2 x=9.38e-04 | v=1.14e-02 | a=1.61e-01
[PHYSICS RAMP: w=1e-03] ep= 3000/4000 | lr=1.06e-04 | total=1.62e-05 | data=7.09e-07 | phys=1.55e-02 | ic=5.07e-09 | relL2 x=8.40e-04 | v=1.04e-02 | a=1.48e-01
  Gradient norms | data=2.57e-04 | phys=5.41e+00 | ic=6.85e-04 | phys/data=2.10e+04
[PHYSICS RAMP: w=1e-03] ep= 3500/4000 | lr=7.27e-05 | total=1.43e-05 | data=5.96e-07 | phys=1.37e-02 | ic=4.96e-09 | relL2 x=7.71e-04 | v=9.58e-03 | a=1.38e-01
[PHYSICS RAMP: w=1e-03] ep= 4000/4000 | lr=5.00e-05 | total=1.34e-05 | data=5.14e-07 | phys=1.29e-02 | ic=5.32e-09 | relL2 x=7.18e-04 | v=8.98e-03 | a=1.30e-01
  Gradient norms | data=2.13e-04 | phys=5.54e+00 | ic=6.37e-04 | phys/data=2.60e+04
After physics weight 1e-03: relL2 x=7.1834e-04 | v=8.9825e-03 | a=1.3016e-01

==============================================================================
PHYSICS RAMP: w=1e-02
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e-02
==============================================================================
[PHYSICS RAMP: w=1e-02] ep=  500/4000 | lr=2.06e-04 | total=1.96e-04 | data=5.74e-06 | phys=1.90e-02 | ic=1.84e-07 | relL2 x=2.40e-03 | v=1.48e-02 | a=1.67e-01
[PHYSICS RAMP: w=1e-02] ep= 1000/4000 | lr=1.42e-04 | total=1.18e-04 | data=2.77e-06 | phys=1.15e-02 | ic=2.50e-07 | relL2 x=1.67e-03 | v=1.14e-02 | a=1.34e-01
  Gradient norms | data=5.53e-04 | phys=6.06e+00 | ic=2.04e-03 | phys/data=1.10e+04
[PHYSICS RAMP: w=1e-02] ep= 1500/4000 | lr=9.76e-05 | total=8.68e-05 | data=1.80e-06 | phys=8.49e-03 | ic=6.26e-08 | relL2 x=1.35e-03 | v=9.80e-03 | a=1.19e-01
[PHYSICS RAMP: w=1e-02] ep= 2000/4000 | lr=6.71e-05 | total=7.33e-05 | data=1.32e-06 | phys=7.20e-03 | ic=4.25e-08 | relL2 x=1.15e-03 | v=8.89e-03 | a=1.10e-01
  Gradient norms | data=3.52e-04 | phys=4.37e+00 | ic=1.19e-03 | phys/data=1.24e+04
[PHYSICS RAMP: w=1e-02] ep= 2500/4000 | lr=4.61e-05 | total=5.80e-05 | data=1.05e-06 | phys=5.70e-03 | ic=2.33e-08 | relL2 x=1.03e-03 | v=8.28e-03 | a=1.04e-01
[PHYSICS RAMP: w=1e-02] ep= 3000/4000 | lr=3.17e-05 | total=5.58e-05 | data=8.83e-07 | phys=5.48e-03 | ic=3.90e-08 | relL2 x=9.42e-04 | v=7.87e-03 | a=9.97e-02
  Gradient norms | data=3.13e-04 | phys=3.26e+00 | ic=1.65e-03 | phys/data=1.04e+04
[PHYSICS RAMP: w=1e-02] ep= 3500/4000 | lr=2.18e-05 | total=4.79e-05 | data=7.58e-07 | phys=4.71e-03 | ic=3.08e-08 | relL2 x=8.72e-04 | v=7.56e-03 | a=9.66e-02
[PHYSICS RAMP: w=1e-02] ep= 4000/4000 | lr=1.50e-05 | total=4.65e-05 | data=6.72e-07 | phys=4.58e-03 | ic=2.66e-08 | relL2 x=8.21e-04 | v=7.32e-03 | a=9.42e-02
  Gradient norms | data=2.09e-04 | phys=2.45e+00 | ic=7.64e-04 | phys/data=1.17e+04
After physics weight 1e-02: relL2 x=8.2112e-04 | v=7.3208e-03 | a=9.4211e-02

==============================================================================
PHYSICS RAMP: w=1e-01
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e-01
==============================================================================
[PHYSICS RAMP: w=1e-01] ep=  500/4000 | lr=2.06e-04 | total=1.08e-02 | data=8.17e-05 | phys=1.07e-01 | ic=2.35e-06 | relL2 x=8.92e-03 | v=1.68e-02 | a=1.32e-01
[PHYSICS RAMP: w=1e-01] ep= 1000/4000 | lr=1.42e-04 | total=1.93e-03 | data=1.09e-05 | phys=1.92e-02 | ic=5.76e-07 | relL2 x=3.29e-03 | v=1.04e-02 | a=9.83e-02
  Gradient norms | data=4.33e-03 | phys=4.72e+01 | ic=3.54e-03 | phys/data=1.09e+04
[PHYSICS RAMP: w=1e-01] ep= 1500/4000 | lr=9.76e-05 | total=9.20e-04 | data=4.11e-06 | phys=9.15e-03 | ic=5.55e-07 | relL2 x=2.04e-03 | v=9.05e-03 | a=9.06e-02
[PHYSICS RAMP: w=1e-01] ep= 2000/4000 | lr=6.71e-05 | total=2.62e-04 | data=2.65e-06 | phys=2.59e-03 | ic=3.41e-07 | relL2 x=1.63e-03 | v=8.30e-03 | a=8.61e-02
  Gradient norms | data=6.54e-04 | phys=5.10e+00 | ic=3.22e-03 | phys/data=7.80e+03
[PHYSICS RAMP: w=1e-01] ep= 2500/4000 | lr=4.61e-05 | total=2.13e-04 | data=2.21e-06 | phys=2.11e-03 | ic=2.55e-07 | relL2 x=1.49e-03 | v=7.90e-03 | a=8.38e-02
[PHYSICS RAMP: w=1e-01] ep= 3000/4000 | lr=3.17e-05 | total=1.95e-04 | data=1.94e-06 | phys=1.92e-03 | ic=2.14e-07 | relL2 x=1.40e-03 | v=7.60e-03 | a=8.21e-02
  Gradient norms | data=4.94e-04 | phys=2.41e+00 | ic=2.99e-03 | phys/data=4.87e+03
[PHYSICS RAMP: w=1e-01] ep= 3500/4000 | lr=2.18e-05 | total=1.65e-04 | data=1.75e-06 | phys=1.63e-03 | ic=1.82e-07 | relL2 x=1.33e-03 | v=7.40e-03 | a=8.09e-02
[PHYSICS RAMP: w=1e-01] ep= 4000/4000 | lr=1.50e-05 | total=1.53e-04 | data=1.63e-06 | phys=1.52e-03 | ic=1.62e-07 | relL2 x=1.28e-03 | v=7.27e-03 | a=8.02e-02
  Gradient norms | data=4.00e-04 | phys=1.83e+00 | ic=2.22e-03 | phys/data=4.57e+03
After physics weight 1e-01: relL2 x=1.2777e-03 | v=7.2664e-03 | a=8.0201e-02

==============================================================================
PHYSICS RAMP: w=1e+00
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e+00
==============================================================================
[PHYSICS RAMP: w=1e+00] ep=  500/4000 | lr=2.06e-04 | total=6.74e-02 | data=5.46e-03 | phys=6.19e-02 | ic=1.54e-05 | relL2 x=7.38e-02 | v=7.47e-02 | a=1.52e-01
[PHYSICS RAMP: w=1e+00] ep= 1000/4000 | lr=1.42e-04 | total=2.03e-02 | data=1.17e-03 | phys=1.91e-02 | ic=8.57e-06 | relL2 x=3.41e-02 | v=3.56e-02 | a=1.05e-01
  Gradient norms | data=6.69e-02 | phys=3.66e+01 | ic=1.62e-02 | phys/data=5.47e+02
[PHYSICS RAMP: w=1e+00] ep= 1500/4000 | lr=9.76e-05 | total=1.14e-02 | data=3.80e-04 | phys=1.11e-02 | ic=6.23e-06 | relL2 x=1.94e-02 | v=2.17e-02 | a=9.44e-02
[PHYSICS RAMP: w=1e+00] ep= 2000/4000 | lr=6.71e-05 | total=4.62e-03 | data=1.54e-04 | phys=4.46e-03 | ic=3.57e-06 | relL2 x=1.24e-02 | v=1.51e-02 | a=8.72e-02
  Gradient norms | data=1.90e-02 | phys=2.24e+01 | ic=1.13e-02 | phys/data=1.18e+03
[PHYSICS RAMP: w=1e+00] ep= 2500/4000 | lr=4.61e-05 | total=2.57e-03 | data=6.90e-05 | phys=2.50e-03 | ic=2.59e-06 | relL2 x=8.28e-03 | v=1.16e-02 | a=8.32e-02
[PHYSICS RAMP: w=1e+00] ep= 3000/4000 | lr=3.17e-05 | total=1.56e-03 | data=3.90e-05 | phys=1.52e-03 | ic=2.56e-06 | relL2 x=6.23e-03 | v=1.01e-02 | a=8.15e-02
  Gradient norms | data=7.60e-03 | phys=8.95e+00 | ic=9.03e-03 | phys/data=1.18e+03
[PHYSICS RAMP: w=1e+00] ep= 3500/4000 | lr=2.18e-05 | total=1.02e-03 | data=2.44e-05 | phys=9.95e-04 | ic=2.17e-06 | relL2 x=4.93e-03 | v=9.28e-03 | a=8.09e-02
[PHYSICS RAMP: w=1e+00] ep= 4000/4000 | lr=1.50e-05 | total=9.64e-04 | data=1.74e-05 | phys=9.45e-04 | ic=1.98e-06 | relL2 x=4.17e-03 | v=8.84e-03 | a=8.04e-02
  Gradient norms | data=4.25e-03 | phys=4.64e+00 | ic=7.88e-03 | phys/data=1.09e+03
After physics weight 1e+00: relL2 x=4.1652e-03 | v=8.8424e-03 | a=8.0410e-02

TRAINING COMPLETE
Selected checkpoint (physics weight 0.01): /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/healthy_fourier_pinn_lrdecay.keras
Final-stage checkpoint: /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/healthy_fourier_pinn_lrdecay_final.keras
Save/load round-trip max diff: 0.000000e+00

Saved run config: models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/healthy_fourier_pinn_lrdecay_run_config.json
Saved history: models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/healthy_fourier_pinn_lrdecay_history.json
Selected checkpoint: /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/healthy_fourier_pinn_lrdecay.keras
Selected metrics: {'x': 0.0008211242600932792, 'v': 0.007320777733201848, 'a': 0.09421143851788975}
(.venv) manjeet_kumar@DESKTOP-NV4RH6T:/mnt/k/Projects/PINNS$ python train_healthy2.py \
  --w-max 5 \
  --n-freq 32 \
  --seed 42 \
  --output-root models1/experiments/optional_checks
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792160875.022402    7457 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792160876.406185    7457 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1792160896.430451    7457 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1792160916.698875    7457 gpu_device.cc:2043] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 3537 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4050 Laptop GPU, pci bus id: 0000:01:00.0, compute capability: 8.9

========== REPRODUCIBLE HEALTHY PINN RUN ==========
seed=42 | w_max=5 | n_freq=32 | width=128 | depth=4
Frequency minimum=0.25; frequency spacing=0.15322581
scales: {'T0': 0.21157993442837367, 'M0': 1.0, 'X0': 0.3941715948658804}
output directory: models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights

==============================================================================
WARM-UP: DATA + IC
Steps: 10000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 0.00e+00
==============================================================================
[WARM-UP: DATA + IC] ep=  500/10000 | lr=8.61e-04 | total=3.78e-04 | data=3.77e-04 | phys=0.00e+00 | ic=1.09e-06 | relL2 x=1.94e-02 | v=2.15e-01 | a=2.69e+00
[WARM-UP: DATA + IC] ep= 1000/10000 | lr=7.41e-04 | total=1.16e-04 | data=1.16e-04 | phys=0.00e+00 | ic=1.21e-08 | relL2 x=1.08e-02 | v=1.43e-01 | a=2.02e+00
  Gradient norms | data=1.11e-03 | phys=1.04e+02 | ic=8.88e-04 | phys/data=9.36e+04
[WARM-UP: DATA + IC] ep= 1500/10000 | lr=6.38e-04 | total=6.01e-05 | data=6.01e-05 | phys=0.00e+00 | ic=5.59e-09 | relL2 x=7.79e-03 | v=1.14e-01 | a=1.71e+00
[WARM-UP: DATA + IC] ep= 2000/10000 | lr=5.49e-04 | total=3.98e-05 | data=3.97e-05 | phys=0.00e+00 | ic=3.15e-09 | relL2 x=6.34e-03 | v=9.71e-02 | a=1.51e+00
  Gradient norms | data=5.42e-04 | phys=7.05e+01 | ic=5.22e-04 | phys/data=1.30e+05
[WARM-UP: DATA + IC] ep= 2500/10000 | lr=4.73e-04 | total=2.67e-05 | data=2.67e-05 | phys=0.00e+00 | ic=1.81e-08 | relL2 x=5.19e-03 | v=8.39e-02 | a=1.36e+00
[WARM-UP: DATA + IC] ep= 3000/10000 | lr=4.07e-04 | total=1.98e-05 | data=1.98e-05 | phys=0.00e+00 | ic=3.22e-09 | relL2 x=4.48e-03 | v=7.48e-02 | a=1.25e+00
  Gradient norms | data=4.35e-04 | phys=5.78e+01 | ic=3.38e-04 | phys/data=1.33e+05
[WARM-UP: DATA + IC] ep= 3500/10000 | lr=3.50e-04 | total=6.67e-05 | data=1.76e-05 | phys=0.00e+00 | ic=4.91e-05 | relL2 x=4.09e-03 | v=6.82e-02 | a=1.16e+00
[WARM-UP: DATA + IC] ep= 4000/10000 | lr=3.02e-04 | total=4.91e-05 | data=1.32e-05 | phys=0.00e+00 | ic=3.59e-05 | relL2 x=3.67e-03 | v=6.28e-02 | a=1.09e+00
  Gradient norms | data=1.79e-03 | phys=5.12e+01 | ic=1.07e-01 | phys/data=2.86e+04
[WARM-UP: DATA + IC] ep= 4500/10000 | lr=2.60e-04 | total=1.06e-05 | data=1.06e-05 | phys=0.00e+00 | ic=7.57e-09 | relL2 x=3.28e-03 | v=5.84e-02 | a=1.03e+00
[WARM-UP: DATA + IC] ep= 5000/10000 | lr=2.24e-04 | total=9.08e-06 | data=9.06e-06 | phys=0.00e+00 | ic=1.89e-08 | relL2 x=3.03e-03 | v=5.47e-02 | a=9.81e-01
  Gradient norms | data=2.72e-04 | phys=4.45e+01 | ic=2.24e-03 | phys/data=1.64e+05
[WARM-UP: DATA + IC] ep= 5500/10000 | lr=1.93e-04 | total=7.85e-06 | data=7.85e-06 | phys=0.00e+00 | ic=1.04e-09 | relL2 x=2.82e-03 | v=5.16e-02 | a=9.36e-01
[WARM-UP: DATA + IC] ep= 6000/10000 | lr=1.66e-04 | total=7.31e-06 | data=6.95e-06 | phys=0.00e+00 | ic=3.62e-07 | relL2 x=2.64e-03 | v=4.89e-02 | a=8.97e-01
  Gradient norms | data=2.12e-04 | phys=4.06e+01 | ic=4.26e-03 | phys/data=1.91e+05
[WARM-UP: DATA + IC] ep= 6500/10000 | lr=1.43e-04 | total=6.78e-06 | data=6.12e-06 | phys=0.00e+00 | ic=6.62e-07 | relL2 x=2.49e-03 | v=4.65e-02 | a=8.62e-01
[WARM-UP: DATA + IC] ep= 7000/10000 | lr=1.23e-04 | total=1.51e-05 | data=5.59e-06 | phys=0.00e+00 | ic=9.51e-06 | relL2 x=2.41e-03 | v=4.43e-02 | a=8.30e-01
  Gradient norms | data=1.26e-03 | phys=3.72e+01 | ic=4.58e-02 | phys/data=2.95e+04
[WARM-UP: DATA + IC] ep= 7500/10000 | lr=1.06e-04 | total=5.28e-06 | data=4.95e-06 | phys=0.00e+00 | ic=3.28e-07 | relL2 x=2.23e-03 | v=4.24e-02 | a=8.02e-01
[WARM-UP: DATA + IC] ep= 8000/10000 | lr=9.10e-05 | total=4.44e-06 | data=4.44e-06 | phys=0.00e+00 | ic=1.15e-09 | relL2 x=2.12e-03 | v=4.07e-02 | a=7.76e-01
  Gradient norms | data=1.98e-04 | phys=3.50e+01 | ic=2.72e-04 | phys/data=1.77e+05
[WARM-UP: DATA + IC] ep= 8500/10000 | lr=7.84e-05 | total=4.05e-06 | data=4.05e-06 | phys=0.00e+00 | ic=4.77e-09 | relL2 x=2.03e-03 | v=3.92e-02 | a=7.53e-01
[WARM-UP: DATA + IC] ep= 9000/10000 | lr=6.75e-05 | total=3.76e-06 | data=3.71e-06 | phys=0.00e+00 | ic=4.92e-08 | relL2 x=1.94e-03 | v=3.78e-02 | a=7.32e-01
  Gradient norms | data=2.97e-04 | phys=3.31e+01 | ic=1.61e-03 | phys/data=1.12e+05
[WARM-UP: DATA + IC] ep= 9500/10000 | lr=5.81e-05 | total=3.44e-06 | data=3.42e-06 | phys=0.00e+00 | ic=1.76e-08 | relL2 x=1.86e-03 | v=3.66e-02 | a=7.13e-01
[WARM-UP: DATA + IC] ep=10000/10000 | lr=5.00e-05 | total=3.18e-06 | data=3.17e-06 | phys=0.00e+00 | ic=7.04e-09 | relL2 x=1.79e-03 | v=3.54e-02 | a=6.95e-01
  Gradient norms | data=1.82e-04 | phys=3.14e+01 | ic=9.77e-04 | phys/data=1.73e+05

WARM-UP FINAL RESULTS
relL2 x=1.7920e-03 | v=3.5401e-02 | a=6.9487e-01

Starting PHYSICS RAMP from the warm-up weights.

==============================================================================
PHYSICS RAMP: w=1e-04
Steps: 4000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 1.00e-04
==============================================================================
[PHYSICS RAMP: w=1e-04] ep=  500/4000 | lr=6.88e-04 | total=7.42e-05 | data=1.98e-05 | phys=5.43e-01 | ic=5.35e-08 | relL2 x=4.44e-03 | v=5.34e-02 | a=7.27e-01
[PHYSICS RAMP: w=1e-04] ep= 1000/4000 | lr=4.73e-04 | total=4.06e-05 | data=9.17e-06 | phys=3.14e-01 | ic=5.66e-08 | relL2 x=3.03e-03 | v=3.89e-02 | a=5.60e-01
  Gradient norms | data=6.71e-04 | phys=2.83e+01 | ic=9.09e-04 | phys/data=4.22e+04
[PHYSICS RAMP: w=1e-04] ep= 1500/4000 | lr=3.25e-04 | total=2.93e-05 | data=6.05e-06 | phys=2.32e-01 | ic=5.20e-08 | relL2 x=2.46e-03 | v=3.26e-02 | a=4.82e-01
[PHYSICS RAMP: w=1e-04] ep= 2000/4000 | lr=2.24e-04 | total=2.22e-05 | data=4.60e-06 | phys=1.75e-01 | ic=1.59e-08 | relL2 x=2.15e-03 | v=2.90e-02 | a=4.35e-01
  Gradient norms | data=4.95e-04 | phys=2.01e+01 | ic=1.20e-03 | phys/data=4.06e+04
[PHYSICS RAMP: w=1e-04] ep= 2500/4000 | lr=1.54e-04 | total=1.89e-05 | data=3.76e-06 | phys=1.51e-01 | ic=1.38e-08 | relL2 x=1.94e-03 | v=2.66e-02 | a=4.03e-01
[PHYSICS RAMP: w=1e-04] ep= 3000/4000 | lr=1.06e-04 | total=1.74e-05 | data=3.23e-06 | phys=1.41e-01 | ic=2.91e-08 | relL2 x=1.80e-03 | v=2.48e-02 | a=3.79e-01
  Gradient norms | data=3.92e-04 | phys=1.45e+01 | ic=9.16e-04 | phys/data=3.69e+04
[PHYSICS RAMP: w=1e-04] ep= 3500/4000 | lr=7.27e-05 | total=1.50e-05 | data=2.85e-06 | phys=1.21e-01 | ic=1.54e-08 | relL2 x=1.69e-03 | v=2.35e-02 | a=3.62e-01
[PHYSICS RAMP: w=1e-04] ep= 4000/4000 | lr=5.00e-05 | total=1.38e-05 | data=2.58e-06 | phys=1.12e-01 | ic=1.59e-08 | relL2 x=1.61e-03 | v=2.25e-02 | a=3.48e-01
  Gradient norms | data=3.07e-04 | phys=1.67e+01 | ic=1.13e-03 | phys/data=5.43e+04
After physics weight 1e-04: relL2 x=1.6076e-03 | v=2.2507e-02 | a=3.4776e-01

==============================================================================
PHYSICS RAMP: w=1e-03
Steps: 4000
Starting learning rate: 1.00e-03
Within-stage LR decay: 0.05x (1.00e-03 -> 5.00e-05)
Physics weight: 1.00e-03
==============================================================================
[PHYSICS RAMP: w=1e-03] ep=  500/4000 | lr=6.88e-04 | total=1.19e-04 | data=1.36e-05 | phys=1.05e-01 | ic=6.50e-07 | relL2 x=3.69e-03 | v=3.62e-02 | a=4.32e-01
[PHYSICS RAMP: w=1e-03] ep= 1000/4000 | lr=4.73e-04 | total=6.23e-05 | data=5.65e-06 | phys=5.66e-02 | ic=9.34e-08 | relL2 x=2.37e-03 | v=2.62e-02 | a=3.28e-01
  Gradient norms | data=6.38e-04 | phys=1.32e+01 | ic=2.31e-03 | phys/data=2.06e+04
[PHYSICS RAMP: w=1e-03] ep= 1500/4000 | lr=3.25e-04 | total=4.23e-05 | data=3.69e-06 | phys=3.86e-02 | ic=4.70e-08 | relL2 x=1.93e-03 | v=2.20e-02 | a=2.82e-01
[PHYSICS RAMP: w=1e-03] ep= 2000/4000 | lr=2.24e-04 | total=3.54e-05 | data=2.77e-06 | phys=3.26e-02 | ic=1.86e-08 | relL2 x=1.67e-03 | v=1.96e-02 | a=2.54e-01
  Gradient norms | data=4.56e-04 | phys=9.43e+00 | ic=1.07e-03 | phys/data=2.07e+04
[PHYSICS RAMP: w=1e-03] ep= 2500/4000 | lr=1.54e-04 | total=2.88e-05 | data=2.29e-06 | phys=2.65e-02 | ic=3.87e-08 | relL2 x=1.51e-03 | v=1.80e-02 | a=2.35e-01
[PHYSICS RAMP: w=1e-03] ep= 3000/4000 | lr=1.06e-04 | total=2.34e-05 | data=1.96e-06 | phys=2.14e-02 | ic=7.53e-08 | relL2 x=1.40e-03 | v=1.69e-02 | a=2.21e-01
  Gradient norms | data=3.68e-04 | phys=8.98e+00 | ic=1.46e-03 | phys/data=2.44e+04
[PHYSICS RAMP: w=1e-03] ep= 3500/4000 | lr=7.27e-05 | total=2.15e-05 | data=1.73e-06 | phys=1.97e-02 | ic=3.74e-08 | relL2 x=1.32e-03 | v=1.61e-02 | a=2.11e-01
[PHYSICS RAMP: w=1e-03] ep= 4000/4000 | lr=5.00e-05 | total=1.94e-05 | data=1.58e-06 | phys=1.78e-02 | ic=1.80e-08 | relL2 x=1.26e-03 | v=1.54e-02 | a=2.03e-01
  Gradient norms | data=3.13e-04 | phys=7.35e+00 | ic=1.01e-03 | phys/data=2.35e+04
After physics weight 1e-03: relL2 x=1.2573e-03 | v=1.5414e-02 | a=2.0323e-01

==============================================================================
PHYSICS RAMP: w=1e-02
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e-02
==============================================================================
[PHYSICS RAMP: w=1e-02] ep=  500/4000 | lr=2.06e-04 | total=1.95e-04 | data=6.58e-06 | phys=1.88e-02 | ic=4.70e-07 | relL2 x=2.57e-03 | v=1.93e-02 | a=2.15e-01
[PHYSICS RAMP: w=1e-02] ep= 1000/4000 | lr=1.42e-04 | total=1.30e-04 | data=3.42e-06 | phys=1.26e-02 | ic=1.96e-07 | relL2 x=1.85e-03 | v=1.59e-02 | a=1.84e-01
  Gradient norms | data=6.98e-04 | phys=6.83e+00 | ic=2.85e-03 | phys/data=9.78e+03
[PHYSICS RAMP: w=1e-02] ep= 1500/4000 | lr=9.76e-05 | total=9.84e-05 | data=2.41e-06 | phys=9.59e-03 | ic=9.06e-08 | relL2 x=1.55e-03 | v=1.43e-02 | a=1.69e-01
[PHYSICS RAMP: w=1e-02] ep= 2000/4000 | lr=6.71e-05 | total=8.43e-05 | data=1.93e-06 | phys=8.23e-03 | ic=9.73e-08 | relL2 x=1.39e-03 | v=1.33e-02 | a=1.59e-01
  Gradient norms | data=5.82e-04 | phys=4.73e+00 | ic=1.89e-03 | phys/data=8.13e+03
[PHYSICS RAMP: w=1e-02] ep= 2500/4000 | lr=4.61e-05 | total=6.68e-05 | data=1.61e-06 | phys=6.51e-03 | ic=5.83e-08 | relL2 x=1.27e-03 | v=1.26e-02 | a=1.52e-01
[PHYSICS RAMP: w=1e-02] ep= 3000/4000 | lr=3.17e-05 | total=6.83e-05 | data=1.42e-06 | phys=6.68e-03 | ic=8.71e-08 | relL2 x=1.19e-03 | v=1.21e-02 | a=1.47e-01
  Gradient norms | data=3.22e-04 | phys=4.31e+00 | ic=2.07e-03 | phys/data=1.34e+04
[PHYSICS RAMP: w=1e-02] ep= 3500/4000 | lr=2.18e-05 | total=5.56e-05 | data=1.28e-06 | phys=5.42e-03 | ic=7.55e-08 | relL2 x=1.14e-03 | v=1.17e-02 | a=1.43e-01
[PHYSICS RAMP: w=1e-02] ep= 4000/4000 | lr=1.50e-05 | total=5.43e-05 | data=1.19e-06 | phys=5.31e-03 | ic=7.32e-08 | relL2 x=1.09e-03 | v=1.14e-02 | a=1.39e-01
  Gradient norms | data=3.55e-04 | phys=3.64e+00 | ic=1.92e-03 | phys/data=1.03e+04
After physics weight 1e-02: relL2 x=1.0944e-03 | v=1.1427e-02 | a=1.3933e-01

==============================================================================
PHYSICS RAMP: w=1e-01
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e-01
==============================================================================
[PHYSICS RAMP: w=1e-01] ep=  500/4000 | lr=2.06e-04 | total=6.98e-03 | data=3.65e-05 | phys=6.94e-02 | ic=1.67e-06 | relL2 x=6.27e-03 | v=1.56e-02 | a=1.44e-01
[PHYSICS RAMP: w=1e-01] ep= 1000/4000 | lr=1.42e-04 | total=3.24e-03 | data=7.83e-06 | phys=3.23e-02 | ic=5.41e-07 | relL2 x=2.80e-03 | v=1.24e-02 | a=1.28e-01
  Gradient norms | data=5.30e-03 | phys=6.92e+01 | ic=4.84e-03 | phys/data=1.30e+04
[PHYSICS RAMP: w=1e-01] ep= 1500/4000 | lr=9.76e-05 | total=9.38e-04 | data=3.01e-06 | phys=9.35e-03 | ic=4.58e-07 | relL2 x=1.74e-03 | v=1.11e-02 | a=1.19e-01
[PHYSICS RAMP: w=1e-01] ep= 2000/4000 | lr=6.71e-05 | total=4.77e-04 | data=2.21e-06 | phys=4.75e-03 | ic=4.46e-07 | relL2 x=1.49e-03 | v=1.04e-02 | a=1.14e-01
  Gradient norms | data=1.22e-03 | phys=2.31e+01 | ic=4.28e-03 | phys/data=1.89e+04
[PHYSICS RAMP: w=1e-01] ep= 2500/4000 | lr=4.61e-05 | total=2.31e-04 | data=1.70e-06 | phys=2.29e-03 | ic=4.88e-07 | relL2 x=1.31e-03 | v=9.85e-03 | a=1.10e-01
[PHYSICS RAMP: w=1e-01] ep= 3000/4000 | lr=3.17e-05 | total=2.16e-04 | data=1.52e-06 | phys=2.15e-03 | ic=3.89e-07 | relL2 x=1.24e-03 | v=9.50e-03 | a=1.08e-01
  Gradient norms | data=4.68e-04 | phys=3.32e+00 | ic=3.99e-03 | phys/data=7.10e+03
[PHYSICS RAMP: w=1e-01] ep= 3500/4000 | lr=2.18e-05 | total=1.87e-04 | data=1.38e-06 | phys=1.86e-03 | ic=3.42e-07 | relL2 x=1.18e-03 | v=9.24e-03 | a=1.05e-01
[PHYSICS RAMP: w=1e-01] ep= 4000/4000 | lr=1.50e-05 | total=1.73e-04 | data=1.29e-06 | phys=1.72e-03 | ic=3.12e-07 | relL2 x=1.14e-03 | v=9.04e-03 | a=1.04e-01
  Gradient norms | data=4.00e-04 | phys=2.13e+00 | ic=3.57e-03 | phys/data=5.33e+03
After physics weight 1e-01: relL2 x=1.1371e-03 | v=9.0410e-03 | a=1.0370e-01

==============================================================================
PHYSICS RAMP: w=1e+00
Steps: 4000
Starting learning rate: 3.00e-04
Within-stage LR decay: 0.05x (3.00e-04 -> 1.50e-05)
Physics weight: 1.00e+00
==============================================================================
[PHYSICS RAMP: w=1e+00] ep=  500/4000 | lr=2.06e-04 | total=4.76e-02 | data=1.92e-03 | phys=4.57e-02 | ic=9.12e-06 | relL2 x=4.36e-02 | v=4.71e-02 | a=1.27e-01
[PHYSICS RAMP: w=1e+00] ep= 1000/4000 | lr=1.42e-04 | total=2.08e-02 | data=4.24e-04 | phys=2.04e-02 | ic=4.08e-06 | relL2 x=2.06e-02 | v=2.36e-02 | a=1.05e-01
  Gradient norms | data=4.95e-02 | phys=5.16e+01 | ic=1.02e-02 | phys/data=1.04e+03
[PHYSICS RAMP: w=1e+00] ep= 1500/4000 | lr=9.76e-05 | total=1.10e-02 | data=9.34e-05 | phys=1.09e-02 | ic=3.95e-06 | relL2 x=9.66e-03 | v=1.35e-02 | a=9.64e-02
[PHYSICS RAMP: w=1e+00] ep= 2000/4000 | lr=6.71e-05 | total=3.31e-03 | data=3.41e-05 | phys=3.28e-03 | ic=1.52e-06 | relL2 x=5.86e-03 | v=1.03e-02 | a=9.22e-02
  Gradient norms | data=1.05e-02 | phys=1.84e+01 | ic=8.16e-03 | phys/data=1.74e+03
[PHYSICS RAMP: w=1e+00] ep= 2500/4000 | lr=4.61e-05 | total=2.71e-03 | data=1.42e-05 | phys=2.70e-03 | ic=2.03e-06 | relL2 x=3.76e-03 | v=9.00e-03 | a=9.00e-02
[PHYSICS RAMP: w=1e+00] ep= 3000/4000 | lr=3.17e-05 | total=1.43e-03 | data=9.48e-06 | phys=1.42e-03 | ic=1.29e-06 | relL2 x=3.09e-03 | v=8.52e-03 | a=8.86e-02
  Gradient norms | data=4.22e-03 | phys=1.05e+01 | ic=7.14e-03 | phys/data=2.49e+03
[PHYSICS RAMP: w=1e+00] ep= 3500/4000 | lr=2.18e-05 | total=9.62e-04 | data=6.35e-06 | phys=9.54e-04 | ic=1.03e-06 | relL2 x=2.52e-03 | v=8.19e-03 | a=8.74e-02
[PHYSICS RAMP: w=1e+00] ep= 4000/4000 | lr=1.50e-05 | total=8.67e-04 | data=4.93e-06 | phys=8.61e-04 | ic=1.06e-06 | relL2 x=2.22e-03 | v=7.98e-03 | a=8.65e-02
  Gradient norms | data=2.43e-03 | phys=4.73e+00 | ic=6.28e-03 | phys/data=1.94e+03
After physics weight 1e+00: relL2 x=2.2197e-03 | v=7.9758e-03 | a=8.6494e-02

TRAINING COMPLETE
Selected checkpoint (physics weight 0.01): /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights/healthy_fourier_pinn_lrdecay.keras
Final-stage checkpoint: /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights/healthy_fourier_pinn_lrdecay_final.keras
Save/load round-trip max diff: 0.000000e+00

Saved run config: models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights/healthy_fourier_pinn_lrdecay_run_config.json
Saved history: models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights/healthy_fourier_pinn_lrdecay_history.json
Selected checkpoint: /mnt/k/Projects/PINNS/models1/experiments/optional_checks/seed_42_wmax_5_nfreq_32/weights/healthy_fourier_pinn_lrdecay.keras
Selected metrics: {'x': 0.0010944160710738492, 'v': 0.011427420485292907, 'a': 0.13932644467589617}
'''