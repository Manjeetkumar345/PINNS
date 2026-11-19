# import numpy as np, tensorflow as tf
# from scipy.linalg import eigh
# from tensorflow import keras
# from tensorflow.keras import layers
# from models import calculate_response_derivatives
# from datasets import prepare_healthy_pinn_data

# keras.backend.set_floatx("float64")      # before building any model

# def build_scaled_problem(case, K_ref):
#     M, C, K = (np.asarray(case[k], float) for k in ["mass_matrix","damping_matrix","stiffness_matrix"])
#     F = np.asarray(case["force_history"], float)
#     x, v, a = (np.asarray(case[k], float) for k in ["displacement","velocity","acceleration"])
#     t = np.asarray(case["time"], float).ravel()
#     w1 = np.sqrt(eigh(K_ref, M, eigvals_only=True)[0])
#     T0, M0, X0 = 1.0/w1, np.mean(np.diag(M)), np.sqrt(np.mean(x**2))
#     return dict(
#         tau=(t/T0).reshape(-1,1), x=x/X0, v=v*T0/X0, a=a*T0**2/X0,
#         M=M/M0, C=C*T0/M0, K=K*T0**2/M0, F=F*T0**2/(M0*X0),
#         T0=T0, X0=X0, M0=M0, tau_max=t[-1]/T0)

# class FourierPINN(keras.Model):
#     def __init__(self, dof, tau_max, w_max=14.0, n_freq=56, width=64, depth=4):
#         super().__init__()
#         self.tau_max = tau_max
#         self.freqs = tf.constant(np.linspace(0.25, w_max, n_freq).reshape(1,-1))  # float64
#         self.body = keras.Sequential(
#             [layers.Dense(width, activation="tanh") for _ in range(depth)] + [layers.Dense(dof)])
#     def call(self, tau):
#         arg = tau * self.freqs
#         return self.body(tf.concat([tf.sin(arg), tf.cos(arg), tau/self.tau_max], axis=1))

# def make_losses(model, P, idx_data, w, n_col=1024):
#     tau_all = tf.constant(P["tau"])
#     xt = tf.constant(P["x"])
#     M, C, K, F = (tf.constant(P[k]) for k in ["M", "C", "K", "F"])
#     f_scale = tf.reduce_mean(F**2)
#     N = tau_all.shape[0]

#     def compute():
#         ci = tf.random.uniform([n_col], 0, N, dtype=tf.int32)
#         tc, Fc = tf.gather(tau_all, ci), tf.gather(F, ci)
#         x_h, v_h, a_h = calculate_response_derivatives(model, tc)
#         R = a_h @ tf.transpose(M) + v_h @ tf.transpose(C) + x_h @ tf.transpose(K) - Fc
#         L_phys = tf.reduce_mean(R**2) / f_scale

#         td = tf.gather(tau_all, idx_data)
#         L_data = tf.reduce_mean((model(td) - tf.gather(xt, idx_data))**2)

#         z = tf.zeros((1, 1), tf.float64)
#         x0, v0, _ = calculate_response_derivatives(model, z)
#         L_ic = tf.reduce_mean(x0**2) + tf.reduce_mean(v0**2)
#         total = w["data"]*L_data + w["phys"]*L_phys + w["ic"]*L_ic
#         return total, L_data, L_phys, L_ic
#     return compute

# case_healthy = prepare_healthy_pinn_data()

# P = build_scaled_problem(case_healthy, K_ref=case_healthy["stiffness_matrix"])
# model = FourierPINN(10, P["tau_max"]); model(tf.constant(P["tau"][:1]))   # build
# w = {k: tf.Variable(1.0, dtype=tf.float64) for k in ["data","phys","ic"]}
# idx = tf.constant(np.arange(0, len(P["tau"]), 4))
# compute = make_losses(model, P, idx, w)

# lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 2000, 0.7)
# # opt = keras.optimizers.Adam(lr)
# w["phys"].assign(0.0)
# opt = keras.optimizers.Adam(1e-3) 

# @tf.function
# def step():
#     with tf.GradientTape() as tape:
#         total, Ld, Lp, Li = compute()
#     g = tape.gradient(total, model.trainable_variables)
#     opt.apply_gradients(zip(g, model.trainable_variables))
#     return total, Ld, Lp, Li

# def derivatives_in_batches(model, tau, batch_size=512):
#     x_parts, v_parts, a_parts = [], [], []

#     for start in range(0, len(tau), batch_size):
#         tau_batch = tf.convert_to_tensor(
#             tau[start:start + batch_size],
#             dtype=tf.float64
#         )

#         x_batch, v_batch, a_batch = calculate_response_derivatives(
#             model, tau_batch
#         )

#         x_parts.append(x_batch.numpy())
#         v_parts.append(v_batch.numpy())
#         a_parts.append(a_batch.numpy())

#     return (
#         np.concatenate(x_parts, axis=0),
#         np.concatenate(v_parts, axis=0),
#         np.concatenate(a_parts, axis=0)
#     )

# for ep in range(20000):

#     total, Ld, Lp, Li = step()

#     if ep % 500 == 0:
#         xh, vh, ah = derivatives_in_batches(
#             model, P["tau"], batch_size=512
#         )

#         rel = lambda p, q: (
#             np.linalg.norm(p - q) / np.linalg.norm(q)
#         )

#         print(
#             ep,
#             f"data {Ld.numpy():.2e} "
#             f"phys {Lp.numpy():.2e} "
#             f"ic {Li.numpy():.2e} | "
#             f"relL2 x {rel(xh, P['x']):.2e} "
#             f"v {rel(vh, P['v']):.2e} "
#             f"a {rel(ah, P['a']):.2e}"
#         )


import numpy as np
import tensorflow as tf

from scipy.linalg import eigh
from tensorflow import keras
from tensorflow.keras import layers

from models import calculate_response_derivatives
from datasets import prepare_healthy_pinn_data,prepare_damaged_pinn_data,generate_case
from structure import masses, stiffness, damping, forces, N

# ============================================================
# 0. CONFIGURATION
# ============================================================

keras.backend.set_floatx("float64")
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

DOF = 10
N_COL = 1024

TEST_A_STEPS = 10000
TEST_B_STEPS_PER_STAGE = 4000

LOG_EVERY = 500
GRAD_EVERY = 1000


# ============================================================
# 1. SCALE THE STRUCTURAL PROBLEM
# ============================================================

def build_scaled_problem(case, K_ref):

    M, C, K = (
        np.asarray(case[k], dtype=np.float64)
        for k in [
            "mass_matrix",
            "damping_matrix",
            "stiffness_matrix"
        ]
    )

    F = np.asarray(case["force_history"], dtype=np.float64)

    x, v, a = (
        np.asarray(case[k], dtype=np.float64)
        for k in [
            "displacement",
            "velocity",
            "acceleration"
        ]
    )

    t = np.asarray(case["time"], dtype=np.float64).ravel()

    omega_1 = np.sqrt(
        eigh(K_ref, M, eigvals_only=True)[0]
    )

    T0 = 1.0 / omega_1
    M0 = np.mean(np.diag(M))
    X0 = np.sqrt(np.mean(x**2))

    if X0 < 1e-12:
        raise ValueError(
            "Reference displacement scale is too small."
        )

    return {
        "tau": (t / T0).reshape(-1, 1),
        "x": x / X0,
        "v": v * T0 / X0,
        "a": a * T0**2 / X0,

        "M": M / M0,
        "C": C * T0 / M0,
        "K": K * T0**2 / M0,
        "F": F * T0**2 / (M0 * X0),

        "T0": T0,
        "X0": X0,
        "M0": M0,
        "tau_max": t[-1] / T0
    }


# ============================================================
# 2. FOURIER PINN
#    Changes: w_max = 8, width = 128
# ============================================================

class FourierPINN(keras.Model):

    def __init__(
        self,
        dof,
        tau_max,
        w_max=8.0,
        n_freq=56,
        width=128,
        depth=4
    ):
        super().__init__()

        self.tau_max = tf.constant(
            tau_max, dtype=tf.float64
        )

        self.freqs = tf.constant(
            np.linspace(
                0.25, w_max, n_freq
            ).reshape(1, -1),
            dtype=tf.float64
        )

        self.body = keras.Sequential(
            [
                layers.Dense(
                    width,
                    activation="tanh",
                    dtype=tf.float64
                )
                for _ in range(depth)
            ]
            + [
                layers.Dense(dof, dtype=tf.float64)
            ]
        )

    def call(self, tau):

        tau = tf.cast(tau, tf.float64)

        arg = tau * self.freqs

        features = tf.concat(
            [
                tf.sin(arg),
                tf.cos(arg),
                tau / self.tau_max
            ],
            axis=1
        )

        return self.body(features)


# ============================================================
# 3. LOSS FUNCTIONS
#    Supports an optional time-window curriculum.
# ============================================================

def make_losses(
    model,
    P,
    idx_data,
    w,
    n_col=1024,
    tau_window=None
):

    tau_all = tf.constant(
        P["tau"], dtype=tf.float64
    )

    x_all = tf.constant(
        P["x"], dtype=tf.float64
    )

    M, C, K, F = (
        tf.constant(P[k], dtype=tf.float64)
        for k in ["M", "C", "K", "F"]
    )

    f_scale = tf.maximum(
        tf.reduce_mean(F**2),
        tf.constant(1e-12, dtype=tf.float64)
    )

    N = tau_all.shape[0]

    def compute():

        # ----------------------------------------------------
        # Physics collocation points
        # ----------------------------------------------------

        if tau_window is None:

            candidate_indices = tf.range(
                N, dtype=tf.int32
            )

        else:

            valid_mask = (
                tau_all[:, 0] <= tau_window
            )

            candidate_indices = tf.cast(
                tf.reshape(
                    tf.where(valid_mask), [-1]
                ),
                tf.int32
            )

        n_candidates = tf.shape(candidate_indices)[0]

        local_indices = tf.random.uniform(
            shape=(n_col,),
            minval=0,
            maxval=n_candidates,
            dtype=tf.int32
        )

        ci = tf.gather(
            candidate_indices, local_indices
        )

        tc = tf.gather(tau_all, ci)
        Fc = tf.gather(F, ci)

        x_h, v_h, a_h = calculate_response_derivatives(
            model, tc
        )

        # M*x'' + C*x' + K*x - F = 0
        R = (
            a_h @ tf.transpose(M)
            + v_h @ tf.transpose(C)
            + x_h @ tf.transpose(K)
            - Fc
        )

        L_phys = (
            tf.reduce_mean(R**2) / f_scale
        )

        # ----------------------------------------------------
        # Sparse observation data loss
        # ----------------------------------------------------

        if tau_window is None:

            idx_used = idx_data

        else:

            observed_window_mask = tf.gather(
                tau_all[:, 0] <= tau_window,
                idx_data
            )

            idx_used = tf.boolean_mask(
                idx_data,
                observed_window_mask
            )

        td = tf.gather(tau_all, idx_used)
        xd = tf.gather(x_all, idx_used)

        L_data = tf.reduce_mean(
            (model(td) - xd)**2
        )

        # ----------------------------------------------------
        # Initial-condition loss
        # ----------------------------------------------------

        z = tf.zeros(
            (1, 1), dtype=tf.float64
        )

        x0, v0, _ = calculate_response_derivatives(
            model, z
        )

        L_ic = (
            tf.reduce_mean(x0**2)
            + tf.reduce_mean(v0**2)
        )

        # ----------------------------------------------------
        # Total weighted loss
        # ----------------------------------------------------

        total = (
            w["data"] * L_data
            + w["phys"] * L_phys
            + w["ic"] * L_ic
        )

        return total, L_data, L_phys, L_ic

    return compute


# ============================================================
# 4. LOAD HEALTHY DATA AND BUILD THE MODEL
# ============================================================

case_healthy = prepare_healthy_pinn_data()
case_damaged = prepare_damaged_pinn_data()


P = build_scaled_problem(
    case_healthy,
    K_ref=case_healthy["stiffness_matrix"]
)


# ============================================================
# REQUIRED DIAGNOSTICS — NO TRAINING
# Insert after build_scaled_problem(...)
# ============================================================

# Use the EXACT stiffness vector used to generate case_healthy.
# Replace this example if your actual stiffness values differ.
healthy_stiffness = np.full(DOF, 1000.0, dtype=np.float64)


# ------------------------------------------------------------
# 1. YOUR DAMAGE FUNCTIONS
# ------------------------------------------------------------

def apply_damage1(stiffness, damage_elements, damage_severities):
    """
    Apply stiffness reduction:
    k_damaged[i] = (1 - alpha_i) * k_healthy[i].

    Elements use zero-based indexing.
    Healthy case: damage_elements=[] and damage_severities=[].
    """
    stiffness = np.asarray(stiffness, dtype=float)
    damage_elements = list(damage_elements)
    damage_severities = [float(s) for s in damage_severities]

    if len(damage_elements) != len(damage_severities):
        raise ValueError(
            "damage_elements and damage_severities "
            "must have the same length."
        )

    if len(set(damage_elements)) != len(damage_elements):
        raise ValueError("Duplicate damage elements are not allowed.")

    damaged_stiffness = stiffness.copy()

    for element, severity in zip(
        damage_elements, damage_severities
    ):
        if not isinstance(element, (int, np.integer)):
            raise ValueError("Each damage element must be an integer.")

        if element < 0 or element >= len(stiffness):
            raise ValueError(
                f"Damage element must be between 0 and "
                f"{len(stiffness) - 1}."
            )

        if not (np.isfinite(severity) and 0.0 <= severity < 1.0):
            raise ValueError(
                "Damage severity must be finite and in [0, 1)."
            )

        damaged_stiffness[element] *= (1.0 - severity)

    return damaged_stiffness


def damage_to_alpha(
    n_elements, damage_elements, damage_severities
):
    """Return the ground-truth damage vector."""
    alpha = np.zeros(n_elements, dtype=np.float64)

    if len(damage_elements) != len(damage_severities):
        raise ValueError("Damage elements and severities must match.")

    for e, s in zip(damage_elements, damage_severities):
        alpha[int(e)] = float(s)

    return alpha


# ------------------------------------------------------------
# 2. ASSEMBLE THE STIFFNESS MATRIX
#    Convention: element 0 connects DOF 0 to ground;
#    element i connects DOF i-1 to DOF i.
# ------------------------------------------------------------

def element_matrices(n_elements):
    E = np.zeros(
        (n_elements, n_elements, n_elements),
        dtype=np.float64
    )

    for i in range(n_elements):
        e = np.zeros(n_elements, dtype=np.float64)
        e[i] = 1.0

        if i > 0:
            e[i - 1] = -1.0

        E[i] = np.outer(e, e)

    return E


def assemble_stiffness_matrix(stiffness, E):
    stiffness = np.asarray(stiffness, dtype=np.float64)

    if len(stiffness) != E.shape[0]:
        raise ValueError(
            "Stiffness vector length does not match element count."
        )

    return np.einsum("i,ijk->jk", stiffness, E)


# ------------------------------------------------------------
# 3. CHECK SIMULATOR KINEMATICS
# ------------------------------------------------------------

def relative_l2_np(pred, true):
    return np.linalg.norm(pred - true) / max(
        np.linalg.norm(true), 1e-12
    )


def kinematics_check(P):
    tau = P["tau"][:, 0]

    # Derivatives with respect to dimensionless time tau
    v_from_x = np.gradient(
        P["x"], tau, axis=0, edge_order=2
    )
    a_from_v = np.gradient(
        P["v"], tau, axis=0, edge_order=2
    )

    # Ignore endpoints where numerical differentiation is less reliable
    sl = slice(2, -2)

    results = {
        "v_vs_dx": relative_l2_np(
            P["v"][sl], v_from_x[sl]
        ),
        "a_vs_dv": relative_l2_np(
            P["a"][sl], a_from_v[sl]
        ),
    }

    print("\n========== KINEMATICS CHECK ==========")
    for name, value in results.items():
        print(f"{name}: {value:.4e}")

    return results


# ------------------------------------------------------------
# 4. DAMAGE-SIGNAL TEST USING apply_damage1()
# ------------------------------------------------------------

def damage_signal(P, healthy_stiffness, dalpha=0.10):
    E = element_matrices(len(healthy_stiffness))

    K_healthy = assemble_stiffness_matrix(
        healthy_stiffness, E
    )

    # Verify that this element convention matches the simulator.
    K_simulator = np.asarray(P["K"]) * P["M0"] / P["T0"]**2

    matrix_error = relative_l2_np(K_healthy, K_simulator)

    print("\n========== STIFFNESS MATRIX CHECK ==========")
    print(f"Relative matrix mismatch: {matrix_error:.4e}")

    if not np.allclose(K_healthy, K_simulator, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "Assembled stiffness does not match the simulator. "
            "Check element ordering, stiffness values, and assembly "
            "convention before interpreting damage signals."
        )

    signals = []

    for element in range(len(healthy_stiffness)):
        # Damage exactly one element at a time.
        damaged_stiffness = apply_damage1(
            healthy_stiffness,
            damage_elements=[element],
            damage_severities=[dalpha],
        )

        alpha_true = damage_to_alpha(
            len(healthy_stiffness),
            [element],
            [dalpha],
        )

        K_damaged = assemble_stiffness_matrix(
            damaged_stiffness, E
        )

        # Stiffness difference in the scaled PINN equation.
        dK_scaled = (
            (K_healthy - K_damaged)
            * P["T0"]**2 / P["M0"]
        )

        # Preliminary sensitivity proxy, using the healthy response.
        residual_change = P["x"] @ dK_scaled.T

        force_scale = max(
            np.mean(P["F"]**2), 1e-12
        )

        signal = np.mean(residual_change**2) / force_scale

        signals.append(signal)

        assert np.isclose(alpha_true[element], dalpha)

    signals = np.asarray(signals)

    print("\n========== 10% DAMAGE SIGNAL ==========")
    for element, value in enumerate(signals):
        print(f"Element {element}: {value:.4e}")

    return signals


# ------------------------------------------------------------
# 5. RUN ONLY THE REQUIRED DIAGNOSTICS
# ------------------------------------------------------------

kinematics_results = kinematics_check(P)

damage_signals = damage_signal(
    P,
    healthy_stiffness=healthy_stiffness,
    dalpha=0.10,
)

# Check damping only if your simulator has generated a damaged case.
print("\n========== DAMPING CHECK ==========")

if "case_damaged" in globals():
    C_healthy = np.asarray(case_healthy["damping_matrix"])
    C_damaged = np.asarray(case_damaged["damping_matrix"])

    print(
        "np.allclose(C_damaged, C_healthy):",
        np.allclose(C_damaged, C_healthy)
    )
else:
    print(
        "Skipped: case_damaged is not defined. "
        "Generate the damaged case using your simulator, "
        "then compare its damping matrix with the healthy one."
    )


# ======================================================
# LEAST-SQUARES DAMAGE RECOVERY
# ======================================================

def lstsq_alpha(case, healthy_stiffness):
    """
    Estimate element damage severities using the true
    simulated displacement, velocity, and acceleration.

    healthy_stiffness:
        Original, undamaged element stiffness vector.
    """

    M = np.asarray(case["mass_matrix"], dtype=np.float64)
    C = np.asarray(case["damping_matrix"], dtype=np.float64)

    x = np.asarray(case["displacement"], dtype=np.float64)
    v = np.asarray(case["velocity"], dtype=np.float64)
    a = np.asarray(case["acceleration"], dtype=np.float64)
    F = np.asarray(case["force_history"], dtype=np.float64)

    k_healthy = np.asarray(
        healthy_stiffness, dtype=np.float64
    ).reshape(-1)

    N = M.shape[0]

    if len(k_healthy) != N:
        raise ValueError(
            "healthy_stiffness length must match the number of DOFs."
        )

    # Element stiffness matrices: shape (N, N, N)
    E = element_matrices(N)

    # Assemble the original healthy stiffness matrix
    K_healthy = np.einsum(
        "i,ijk->jk",
        k_healthy,
        E
    )

    # Right-hand side:
    # M*a + C*v + K_healthy*x - F
    rhs = (
        a @ M.T
        + v @ C.T
        + x @ K_healthy.T
        - F
    ).reshape(-1)

    # Each column corresponds to one unknown alpha_i
    A = np.stack(
        [
            (
                k_healthy[i] * (x @ E[i].T)
            ).reshape(-1)
            for i in range(N)
        ],
        axis=1
    )

    alpha_hat, *_ = np.linalg.lstsq(
        A,
        rhs,
        rcond=None
    )

    return alpha_hat
# ======================================================
# HEALTHY CASE
# ======================================================

alpha_true_healthy = damage_to_alpha(
    n_elements=len(stiffness),
    damage_elements=[],
    damage_severities=[]
)

alpha_hat_healthy = lstsq_alpha(
    case_healthy,
    healthy_stiffness=stiffness
)

print("\n========== HEALTHY CASE ==========")
print("True alpha:")
print(np.round(alpha_true_healthy, 6))

print("Estimated alpha:")
print(np.round(alpha_hat_healthy, 6))

print(
    "Maximum absolute error:",
    np.max(np.abs(alpha_hat_healthy - alpha_true_healthy))
)


## ======================================================
# DAMAGED CASES: LEAST-SQUARES RECOVERY
# ======================================================

damage_tests = [
    {
        "name": "EASY",
        "damage_elements": [1],
        "damage_severities": [0.30]
    },
    {
        "name": "MEDIUM",
        "damage_elements": [5],
        "damage_severities": [0.20]
    },
    {
        "name": "HARD",
        "damage_elements": [9],
        "damage_severities": [0.20]
    }
]

lstsq_results = {}

for test in damage_tests:

    name = test["name"]
    elements = test["damage_elements"]
    severities = test["damage_severities"]

    print(f"\n========== {name} CASE ==========")

    # --------------------------------------------------
    # 1. Apply your existing damage function
    # --------------------------------------------------

    damaged_stiffness = apply_damage1(
        stiffness=stiffness,
        damage_elements=elements,
        damage_severities=severities
    )

    # --------------------------------------------------
    # 2. Generate the damaged simulation case
    #
    # Pass the ORIGINAL stiffness vector.
    # generate_case should apply the damage internally.
    # --------------------------------------------------

    damaged_case = generate_case(
        masses=masses,
        stiffness=stiffness,
        damping=damping,
        damage_elements=elements,
        damage_severities=severities,
        forces=forces
    )

    # --------------------------------------------------
    # 3. Verify that the generated K matches the
    #    stiffness reduction defined by apply_damage1
    # --------------------------------------------------

    E = element_matrices(len(stiffness))

    K_expected = np.einsum(
        "i,ijk->jk",
        damaged_stiffness,
        E
    )

    K_generated = np.asarray(
        damaged_case["stiffness_matrix"],
        dtype=np.float64
    )

    matrix_match = np.allclose(
        K_expected,
        K_generated,
        rtol=1e-8,
        atol=1e-8
    )

    print("Damaged stiffness matrix matches:", matrix_match)

    if not matrix_match:
        raise ValueError(
            "Generated stiffness matrix does not match "
            "the stiffness produced by apply_damage1()."
        )

    # --------------------------------------------------
    # 4. Ground-truth alpha from your function
    # --------------------------------------------------

    alpha_true = damage_to_alpha(
        n_elements=len(stiffness),
        damage_elements=elements,
        damage_severities=severities
    )

    # --------------------------------------------------
    # 5. Recover alpha using least squares
    # --------------------------------------------------

    alpha_hat = lstsq_alpha(
        case=damaged_case,
        healthy_stiffness=stiffness
    )

    # --------------------------------------------------
    # 6. Evaluate recovery and localization
    # --------------------------------------------------

    absolute_error = np.abs(alpha_hat - alpha_true)

    max_error = np.max(absolute_error)
    mean_error = np.mean(absolute_error)

    predicted_element = int(np.argmax(alpha_hat))

    print("True alpha:     ", np.round(alpha_true, 6))
    print("Estimated alpha:", np.round(alpha_hat, 6))
    print("Absolute error: ", np.round(absolute_error, 6))

    print(f"Maximum absolute alpha error: {max_error:.8e}")
    print(f"Mean absolute alpha error:    {mean_error:.8e}")

    print("True damaged element(s):", elements)
    print("Predicted damaged element:", predicted_element)
    print("Localization correct:", predicted_element in elements)

    lstsq_results[name] = {
        "alpha_true": alpha_true,
        "alpha_hat": alpha_hat,
        "max_error": max_error,
        "mean_error": mean_error,
        "predicted_element": predicted_element,
        "localization_correct": predicted_element in elements
    }


# ======================================================
# LEAST-SQUARES SENSITIVITY TO RESPONSE ERRORS
# ======================================================

def lstsq_sensitivity(
    case,
    healthy_stiffness,
    rel_err=1e-2,
    seed=0
):
    """
    Test damage recovery when displacement, velocity,
    and acceleration contain independently perturbed noise.

    rel_err=1e-3 : small perturbation
    rel_err=1e-2 : moderate perturbation
    rel_err=5e-2 : larger perturbation

    Returns the estimated damage vector.
    """

    if rel_err < 0:
        raise ValueError("rel_err must be non-negative.")

    rng = np.random.default_rng(seed)
    perturbed_case = dict(case)

    for key in [
        "displacement",
        "velocity",
        "acceleration"
    ]:
        arr = np.asarray(case[key], dtype=np.float64)

        noise = (
            rel_err
            * np.std(arr)
            * rng.standard_normal(arr.shape)
        )

        perturbed_case[key] = arr + noise

    # Use your existing least-squares recovery function.
    return lstsq_alpha(
        perturbed_case,
        healthy_stiffness=healthy_stiffness
    )

# ======================================================
# SENSITIVITY EXPERIMENT
# ======================================================

damage_tests = [
    {
        "name": "EASY",
        "damage_elements": [1],
        "damage_severities": [0.30]
    },
    {
        "name": "MEDIUM",
        "damage_elements": [5],
        "damage_severities": [0.20]
    },
    {
        "name": "HARD",
        "damage_elements": [9],
        "damage_severities": [0.20]
    }
]

perturbation_levels = [1e-3, 1e-2, 5e-2]

sensitivity_results = {}

for case_idx, test in enumerate(damage_tests):

    name = test["name"]
    elements = test["damage_elements"]
    severities = test["damage_severities"]

    # Generate the damaged case from the original parameters.
    damaged_case = generate_case(
        masses=masses,
        stiffness=stiffness,
        damping=damping,
        damage_elements=elements,
        damage_severities=severities,
        forces=forces
    )

    # Ground-truth damage vector from your function.
    alpha_true = damage_to_alpha(
        n_elements=len(stiffness),
        damage_elements=elements,
        damage_severities=severities
    )

    print(f"\n{'=' * 55}")
    print(f"{name} CASE")
    print("True alpha:  ", np.round(alpha_true, 5))
    print(f"{'=' * 55}")

    sensitivity_results[name] = {}

    for rel_err in perturbation_levels:

        # Same seed within a case across noise levels:
        # the standardized noise pattern remains comparable.
        alpha_hat = lstsq_sensitivity(
            case=damaged_case,
            healthy_stiffness=stiffness,
            rel_err=rel_err,
            seed=100 + case_idx
        )

        abs_error = np.abs(alpha_hat - alpha_true)
        max_error = np.max(abs_error)
        mean_error = np.mean(abs_error)

        predicted_element = int(np.argmax(alpha_hat))

        print(f"\nPerturbation level: {rel_err:g}")
        print("Estimated alpha:", np.round(alpha_hat, 5))
        print("Absolute error: ", np.round(abs_error, 5))
        print(f"Maximum alpha error: {max_error:.6e}")
        print(f"Mean alpha error:    {mean_error:.6e}")
        print("True damaged element(s):", elements)
        print("Predicted element:", predicted_element)
        print(
            "Localization correct:",
            predicted_element in elements
        )

        sensitivity_results[name][rel_err] = {
            "alpha_true": alpha_true.copy(),
            "alpha_hat": alpha_hat.copy(),
            "max_error": max_error,
            "mean_error": mean_error,
            "predicted_element": predicted_element,
            "localization_correct": predicted_element in elements
        }

# ======================================================
# SUMMARY
# ======================================================

print("\n========== SENSITIVITY SUMMARY ==========")
print(
    f"{'Case':<10} {'Noise':>10} "
    f"{'Max |Error|':>16} {'Mean |Error|':>16} "
    f"{'Localization':>15}"
)

for name, results in sensitivity_results.items():
    for rel_err, result in results.items():

        print(
            f"{name:<10} "
            f"{rel_err:>10.1e} "
            f"{result['max_error']:>16.6e} "
            f"{result['mean_error']:>16.6e} "
            f"{str(result['localization_correct']):>15}"
        )

print("\nDiagnostics complete. No PINN training was run.")

