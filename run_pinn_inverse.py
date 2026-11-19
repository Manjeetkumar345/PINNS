"""
run_inverse_experiment.py
=========================
First inverse experiment: element 1, severity 0.30, clean full-field displacement.
Runs lambda_phys = 1e-2 and 1e-1 as two INDEPENDENT runs: each run reloads the
original healthy checkpoint, builds a fresh DamageParameters object and fresh
optimizers, and uses the same seed and data.

Usage:
    python run_inverse_experiment.py
    python run_inverse_experiment.py --lam-phys 1e-2 --seed 42
    python run_inverse_experiment.py --adapt-steps 0        # joint stage only

ADJUST the import lines marked "<- adjust" if your module names differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

keras.backend.set_floatx("float64")          # before any model is built or loaded

from datasets import generate_case                               # <- adjust if needed
from structure import StructuralDynamics                         # <- adjust if needed

from models1 import (
    DamageParameters,
    InverseConfig,
    load_pretrained_checkpoint,
    prepare_inverse_data,
    train_inverse,
    true_alpha,
    verify_case_consistency,
    verify_checkpoint,
    verify_structure_assembly,
    prepare_training_data

)

# ---- structure and excitation (as used in the project) -----------------
N = 10
MASSES = [1] * N
STIFFNESS = [1000] * N
DAMPING = [2] * N
FORCES = [
    {"floor": 0, "amplitude": 10.0, "angular_frequency": 5.0},
    {"floor": 1, "amplitude": 8.0, "angular_frequency": 4.0},
    {"floor": 2, "amplitude": 12.0, "angular_frequency": 6.0},
    {"floor": 4, "amplitude": 7.0, "angular_frequency": 3.0},
    {"floor": 5, "amplitude": 9.0, "angular_frequency": 5.0},
    {"floor": 7, "amplitude": 6.0, "angular_frequency": 2.0},
    {"floor": 9, "amplitude": 11.0, "angular_frequency": 4.0},
]

DEFAULT_CKPT = ("models1/experiments/optional_checks/seed_42_wmax_8_nfreq_56/weights/"
                "healthy_fourier_pinn_lrdecay.keras")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-root", default="models1/inverse_experiments")
    ap.add_argument("--lam-phys", type=float, nargs="+", default=[1e-2, 1e-1])
    ap.add_argument("--element", type=int, default=1)
    ap.add_argument("--severity", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapt-steps", type=int, default=5000)
    ap.add_argument("--joint-steps", type=int, default=12000)
    ap.add_argument("--lr-alpha", type=float, default=1e-2)
    ap.add_argument("--skip-structure-check", action="store_true",
                    help="skip verify_structure_assembly (use if StructuralDynamics attribute names differ)")
    args = ap.parse_args()

    k_healthy = np.asarray(STIFFNESS, dtype=float)

    # ---- structure assembly check (inverse element matrices vs. your class) ----
    if not args.skip_structure_check:
        structure = StructuralDynamics(masses=MASSES, stiffness=STIFFNESS, damping=DAMPING)
        print("Structure assembly check:", verify_structure_assembly(structure))

    # ---- data: generated with the existing simulator ---------------------------
    case_h = generate_case(MASSES, STIFFNESS, DAMPING, [], [], FORCES)
    case_d = generate_case(MASSES, STIFFNESS, DAMPING, [args.element], [args.severity], FORCES)
    alpha_true = true_alpha(N, [args.element], [args.severity])       # evaluation only
    verify_case_consistency(case_h, case_d, alpha_true, k_healthy)
    print("Case check passed: K(alpha_true) matches the simulator, C unchanged, ODE residual small.")

    summary = {}
    for lam in args.lam_phys:
        tag = f"seed_{args.seed}_el{args.element}_sev{args.severity:g}_lamp{lam:g}"
        out_dir = Path(args.out_root) / tag
        print(f"\n######## RUN {tag} ########")

        keras.utils.set_random_seed(args.seed)                        # same seed for every run

        # Fresh load of the ORIGINAL healthy checkpoint + its required scale sidecar.
        model, scales = load_pretrained_checkpoint(args.ckpt)

        # Healthy scales are reused so the network's output normalization is unchanged.
        P_h = prepare_training_data(case_h, scales=scales)
        P_d = prepare_inverse_data(case_d, scales)                    # true K removed
        verify_checkpoint(model, P_h)

        cfg = InverseConfig(seed=args.seed, lam_phys=lam,
                            adapt_steps=args.adapt_steps, joint_steps=args.joint_steps,
                            lr_alpha=args.lr_alpha)
        dmg = DamageParameters(k_healthy, T0=scales["T0"], M0=scales["M0"],
                               alpha_init=cfg.alpha_init, alpha_max=cfg.alpha_max)

        extra = {
            "initial_checkpoint": str(Path(args.ckpt).resolve()),
            "scales_path": str(Path(args.ckpt).with_name(f"{Path(args.ckpt).stem}_scales.json").resolve()),
            "damage_case": {"elements": [args.element], "severities": [args.severity]},
            "sensors": "all 10 DOFs, clean displacement",
            "excitation": FORCES,
            "masses": MASSES, "stiffness": STIFFNESS, "damping": DAMPING,
        }
        summary[tag] = train_inverse(model, dmg, P_d, alpha_true, cfg, out_dir, scales, extra)
        del model, dmg

    # ---- side-by-side comparison ------------------------------------------------
    print("\n" + "=" * 78 + "\nCOMPARISON (final model of each run)\n" + "=" * 78)
    for tag, r in summary.items():
        d, e = r["final"]["damage"], r["final"]["eval"]
        print(f"{tag}\n"
              f"  argmax={d['argmax_element']}  rank(true)={d['rank_of_true_elements']}  "
              f"alpha_err(true)={d['alpha_error_at_true']}\n"
              f"  max_healthy_alpha={d['max_healthy_alpha']:.3f}  "
              f"n_healthy_above_0.05={d['n_healthy_above_threshold']}\n"
              f"  relL2 x={e['relL2_x']:.2e}  v={e['relL2_v_diag']:.2e}  a={e['relL2_a_diag']:.2e}  "
              f"resid RMSE(scaled)={e['residual_rmse_scaled']:.2e}  "
              f"max={e['residual_max_abs_scaled']:.2e}")
        sel = r["selected"]
        if sel["damage"] is not None:
            print(f"  selected (step {sel['step']}): alpha_err(true)={sel['damage']['alpha_error_at_true']}  "
                  f"max_healthy={sel['damage']['max_healthy_alpha']:.3f}")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    (Path(args.out_root) / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
