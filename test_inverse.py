"""
Run with:  pytest -q test_pinn_inverse.py
Optional real-checkpoint test:
    INVERSE_CKPT=<path to healthy .keras (with *_scales.json beside it)> pytest -q test_pinn_inverse.py
"""
import json
import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

keras.backend.set_floatx("float64")
          
from models1 import (
    FourierPINN ,
    ALPHA_MAX,
    DamageParameters,
    InverseConfig,
    element_matrices,
    evaluate_inverse,
    inverse_losses,
    load_pretrained_checkpoint,
    prepare_inverse_data,
    stiffness_physical,
    train_inverse,
    true_alpha,
    verify_structure_assembly,
)

F64 = tf.float64
N = 4
K_H = np.array([1000.0, 900.0, 1100.0, 1000.0])
T0, M0, X0 = 0.2, 1.0, 0.1
SCALES = {"T0": T0, "M0": M0, "X0": X0}


def reference_K(alpha, k):
    """Independent loop assembly of a fixed-base shear chain."""
    K = np.zeros((len(k), len(k)))
    for i in range(len(k)):
        ke = k[i] * (1.0 - alpha[i])
        K[i, i] += ke
        if i > 0:
            K[i - 1, i - 1] += ke
            K[i - 1, i] -= ke
            K[i, i - 1] -= ke
    return K


def tiny_model():
    model = FourierPINN(dof=N, tau_max=5.0, w_max=3.0, n_freq=6, width=16, depth=2)
    model(tf.zeros((1, 1), F64))
    return model


def tiny_problem(seed=0):
    keras.utils.set_random_seed(seed)
    n_t = 200
    tau = np.linspace(0, 5, n_t).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    x = np.sin(tau * np.arange(1, N + 1)) * 0.1
    P = {
        "tau": tf.constant(tau, F64), "x": tf.constant(x, F64),
        "v": tf.constant(np.cos(tau * np.arange(1, N + 1)) * 0.1, F64),   # diagnostics only
        "a": tf.constant(-np.sin(tau * np.arange(1, N + 1)) * 0.1, F64),  # diagnostics only
        "M": tf.constant(np.eye(N), F64),
        "C": tf.constant(0.01 * np.eye(N), F64),
        "F": tf.constant(rng.standard_normal((n_t, N)), F64),
    }
    model = tiny_model()
    dmg = DamageParameters(K_H, T0=T0, M0=M0, alpha_init=0.02)
    w = {"data": tf.constant(1.0, F64), "phys": tf.constant(1e-2, F64), "ic": tf.constant(1.0, F64)}
    idx = tf.constant(np.arange(0, n_t, 4, dtype=np.int32))
    return model, dmg, P, w, idx


def fake_case(n_t=50, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "mass_matrix": np.eye(N), "damping_matrix": 0.01 * np.eye(N),
        "stiffness_matrix": reference_K(np.zeros(N), K_H),
        "force_history": rng.standard_normal((n_t, N)),
        "displacement": 0.1 * rng.standard_normal((n_t, N)),
        "velocity": 0.1 * rng.standard_normal((n_t, N)),
        "acceleration": 0.1 * rng.standard_normal((n_t, N)),
        "time": np.linspace(0.0, 2.0, n_t),
    }


# ------------------------------------------------------------------
# Stiffness assembly and damage parameters
# ------------------------------------------------------------------

def test_element_matrices_sum_to_reference():
    alpha = np.array([0.0, 0.3, 0.1, 0.0])
    assert np.allclose(stiffness_physical(alpha, K_H), reference_K(alpha, K_H))
    assert element_matrices(N).shape == (N, N, N)


def test_stiffness_assembly_matches_reference_and_scaling():
    _, dmg, *_ = tiny_problem()
    alpha = dmg.alpha().numpy()
    expected = reference_K(alpha, K_H) * (T0 ** 2 / M0)
    assert np.allclose(dmg.stiffness().numpy(), expected, rtol=1e-12)


def test_alpha_init_and_bounds():
    _, dmg, *_ = tiny_problem()
    a = dmg.alpha().numpy()
    assert np.allclose(a, 0.02, atol=1e-12)
    assert np.all(a > 0) and np.all(a < ALPHA_MAX)
    dmg.alpha_raw.assign(np.full(N, 50.0))
    assert np.all(dmg.alpha().numpy() <= ALPHA_MAX)


def test_stiffness_differentiable_wrt_alpha_raw():
    _, dmg, *_ = tiny_problem()
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(dmg.stiffness()))
    g = tape.gradient(loss, dmg.alpha_raw)
    assert g is not None and np.all(np.isfinite(g.numpy())) and np.linalg.norm(g.numpy()) > 0


def test_damage_parameters_validation():
    with pytest.raises(ValueError):
        DamageParameters(K_H, T0=T0, M0=M0, alpha_init=0.7)       # above alpha_max
    with pytest.raises(ValueError):
        DamageParameters(-K_H, T0=T0, M0=M0)                      # non-positive stiffness
    with pytest.raises(ValueError):
        DamageParameters(K_H, T0=0.0, M0=M0)                      # invalid scale


def test_true_alpha_validation():
    assert np.allclose(true_alpha(4, [1], [0.3]), [0, 0.3, 0, 0])
    with pytest.raises(ValueError):
        true_alpha(4, [1, 1], [0.1, 0.2])                         # duplicate element
    with pytest.raises(ValueError):
        true_alpha(4, [4], [0.1])                                 # out of range
    with pytest.raises(ValueError):
        true_alpha(4, [1], [1.0])                                 # severity must be < 1


# ------------------------------------------------------------------
# Gradients and determinism
# ------------------------------------------------------------------

def test_joint_gradients_finite_and_nonzero():
    model, dmg, P, w, idx = tiny_problem()
    with tf.GradientTape() as tape:
        total, *_ = inverse_losses(model, dmg, P, idx, w, n_col=64, include_physics=True)
    grads = tape.gradient(total, model.trainable_variables + [dmg.alpha_raw])
    assert all(g is not None for g in grads), "disconnected gradient"
    assert all(np.all(np.isfinite(g.numpy())) for g in grads)
    assert np.linalg.norm(grads[-1].numpy()) > 0, "alpha gradient is zero"
    assert tf.linalg.global_norm(grads[:-1]).numpy() > 0, "network gradient is zero"


def test_alpha_gets_no_gradient_without_physics():
    model, dmg, P, w, idx = tiny_problem()
    with tf.GradientTape() as tape:
        total, *_ = inverse_losses(model, dmg, P, idx, w, n_col=64, include_physics=False)
    assert tape.gradient(total, dmg.alpha_raw) is None


def test_reproducible_initialization():
    def first_loss():
        model, dmg, P, w, idx = tiny_problem(seed=7)
        return inverse_losses(model, dmg, P, idx, w, n_col=64)[0].numpy()
    assert np.isclose(first_loss(), first_loss(), rtol=0, atol=1e-14)


# ------------------------------------------------------------------
# Data preparation: the true K must never reach training
# ------------------------------------------------------------------

def test_prepare_inverse_data_hides_true_K():
    case = fake_case()
    P = prepare_inverse_data(case, SCALES)
    assert "K" not in P and "stiffness_matrix" not in P
    for key in ("tau", "x", "v", "a", "M", "C", "F"):
        assert key in P
        assert P[key].dtype == tf.float64
    assert np.allclose(P["tau"].numpy().ravel(), case["time"] / T0)
    assert np.allclose(P["x"].numpy(), case["displacement"] / X0)   # healthy scales reused, not recomputed


def test_train_inverse_rejects_true_K(tmp_path):
    model, dmg, P, *_ = tiny_problem()
    bad = dict(P)
    bad["K"] = tf.constant(np.eye(N), F64)
    with pytest.raises(ValueError):
        train_inverse(model, dmg, bad, np.zeros(N), InverseConfig(adapt_steps=1, joint_steps=1),
                      tmp_path, SCALES)


# ------------------------------------------------------------------
# Checkpoint and sidecar handling
# ------------------------------------------------------------------

def test_load_pretrained_checkpoint_requires_sidecar(tmp_path):
    model = tiny_model()
    path = tmp_path / "tiny.keras"
    model.save(str(path))
    with pytest.raises(FileNotFoundError):
        load_pretrained_checkpoint(path)                          # sidecar missing
    (tmp_path / "tiny_scales.json").write_text(json.dumps(SCALES))
    loaded, scales = load_pretrained_checkpoint(path)
    assert scales == pytest.approx(SCALES)
    tau = tf.constant(np.linspace(0, 5, 7).reshape(-1, 1), F64)
    assert np.allclose(loaded(tau).numpy(), model(tau).numpy(), atol=1e-12)


@pytest.mark.skipif("INVERSE_CKPT" not in os.environ, reason="set INVERSE_CKPT to test the real checkpoint")
def test_real_checkpoint_loads_and_predicts():
    model, scales = load_pretrained_checkpoint(os.environ["INVERSE_CKPT"])
    out = model(tf.zeros((3, 1), F64)).numpy()
    assert out.shape == (3, 10) and np.all(np.isfinite(out))
    assert all(scales[k] > 0 for k in ("T0", "M0", "X0"))


# ------------------------------------------------------------------
# Structure assembly check against a stand-in StructuralDynamics
# ------------------------------------------------------------------

class FakeStructure:
    def __init__(self, masses, stiffness, damping):
        self.masses = np.asarray(masses, float)
        self.stiffness = np.asarray(stiffness, float)
        self.damping = np.asarray(damping, float)

    def build_stiffness_matrix(self):
        return reference_K(np.zeros(len(self.stiffness)), self.stiffness)


def test_verify_structure_assembly_with_stand_in():
    out = verify_structure_assembly(FakeStructure(np.ones(N), K_H, 2 * np.ones(N)))
    assert out["healthy_stiffness_relative_error"] < 1e-10
    assert out["damaged_stiffness_relative_error"] < 1e-10


# ------------------------------------------------------------------
# Short end-to-end run: outputs exist and selected state is paired
# ------------------------------------------------------------------

def test_train_inverse_smoke_and_selected_pairing(tmp_path):
    model, dmg, P, *_ = tiny_problem()
    cfg = InverseConfig(seed=0, lam_phys=1e-2, n_col=32, adapt_steps=3, joint_steps=6,
                        log_every=3, data_stride=4)
    alpha_true = np.array([0.0, 0.3, 0.0, 0.0])
    res = train_inverse(model, dmg, P, alpha_true, cfg, tmp_path, SCALES)

    for name in ("config.json", "history.json", "metrics.json",
                 "inverse_final.keras", "inverse_final_alpha.json", "inverse_final_scales.json",
                 "inverse_selected.keras", "inverse_selected_alpha.json", "inverse_selected_scales.json"):
        assert (tmp_path / name).exists(), name

    hist = json.loads((tmp_path / "history.json").read_text())
    assert all(np.isfinite(v) for v in hist["total"])
    assert len(hist["alpha"][0]) == N

    # The selected model and its saved alpha must reproduce the stored selected-state objective.
    sel_model = keras.models.load_model(
        str(tmp_path / "inverse_selected.keras"),
        custom_objects={"FourierPINN": FourierPINN}, compile=False)
    sel_alpha = np.asarray(json.loads((tmp_path / "inverse_selected_alpha.json").read_text())["alpha"])
    dmg2 = DamageParameters(K_H, T0=T0, M0=M0)
    p = sel_alpha / ALPHA_MAX
    dmg2.alpha_raw.assign(np.log(p / (1.0 - p)))
    ev = evaluate_inverse(sel_model, dmg2, P, cfg.lam_phys, cfg.lam_ic)
    assert ev["objective"] == pytest.approx(res["selected"]["eval"]["objective"], rel=1e-6)