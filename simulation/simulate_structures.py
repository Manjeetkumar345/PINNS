import numpy as np
from scipy.integrate import solve_ivp

# ======================================================
# FORWARD SIMULATION
# ======================================================

def simulate_structure(
    structure,
    t_start=0.0,
    t_end=10.0,
    num_points=8000,
    initial_displacement=None,
    initial_velocity=None
):
    """
    Simulate the structural response.

    Returns:
        time
        displacement
        velocity
        acceleration
    """

    N = structure.dof

    # --------------------------------------------------
    # Default initial conditions
    # --------------------------------------------------

    if initial_displacement is None:

        initial_displacement = np.zeros(N)

    else:

        initial_displacement = np.asarray(
            initial_displacement,
            dtype=float
        )

    if initial_velocity is None:

        initial_velocity = np.zeros(N)

    else:

        initial_velocity = np.asarray(
            initial_velocity,
            dtype=float
        )

    # --------------------------------------------------
    # Validate initial conditions
    # --------------------------------------------------

    if len(initial_displacement) != N:

        raise ValueError(
            "Initial displacement must have "
            f"{N} values."
        )

    if len(initial_velocity) != N:

        raise ValueError(
            "Initial velocity must have "
            f"{N} values."
        )

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    initial_state = np.column_stack(
        (
            initial_displacement,
            initial_velocity
        )
    ).flatten()

    # --------------------------------------------------
    # Time points
    # --------------------------------------------------

    time = np.linspace(
        t_start,
        t_end,
        num_points
    )

    # --------------------------------------------------
    # Solve ODE
    # --------------------------------------------------

    solution = solve_ivp(
        fun=structure.structural_dynamics,
        t_span=(t_start, t_end),
        y0=initial_state,
        t_eval=time,
        method="RK45"
    )

    if not solution.success:

        raise RuntimeError(
            f"Structural simulation failed: "
            f"{solution.message}"
        )

    # --------------------------------------------------
    # Convert solver output
    # --------------------------------------------------

    states = solution.y.T

    # --------------------------------------------------
    # Extract displacement and velocity
    # --------------------------------------------------

    displacement = states[
        :, 0::2
    ]

    velocity = states[
        :, 1::2
    ]

    # --------------------------------------------------
    # Calculate acceleration
    # --------------------------------------------------

    acceleration = np.zeros_like(
        displacement
    )

    M = structure.build_mass_matrix()

    C = structure.build_damping_matrix()

    K = structure.build_stiffness_matrix()

    for i, current_time in enumerate(time):

        F = structure.external_force(
            current_time
        )

        acceleration[i] = np.linalg.solve(
            M,
            F
            - C @ velocity[i]
            - K @ displacement[i]
        )

    force_history = np.array([
    structure.external_force(current_time)
    for current_time in time
    ])

    return (
        time,
        displacement,
        velocity,
        acceleration,
        force_history
    )
