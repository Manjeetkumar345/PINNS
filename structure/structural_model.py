import numpy as np


np.set_printoptions(
    precision=6,
    suppress=True
)


# ======================================================
# STRUCTURAL DYNAMICS
# ======================================================

class StructuralDynamics:

    def __init__(
        self,
        masses,
        stiffness,
        damping
    ):

        self.masses = np.asarray(
            masses,
            dtype=float
        )

        self.stiffness = np.asarray(
            stiffness,
            dtype=float
        )

        self.damping = np.asarray(
            damping,
            dtype=float
        )

        self.dof = len(self.masses)

        # ------------------------------------------------
        # External forces
        # ------------------------------------------------

        self.forces = []

        # ------------------------------------------------
        # Validate dimensions
        # ------------------------------------------------

        if len(self.stiffness) != self.dof:
            raise ValueError(
                "Number of stiffness values must equal "
                "number of masses."
            )

        if len(self.damping) != self.dof:
            raise ValueError(
                "Number of damping values must equal "
                "number of masses."
            )

        # ------------------------------------------------
        # Validate physical parameters
        # ------------------------------------------------

        if np.any(self.masses <= 0):
            raise ValueError(
                "All mass values must be greater than 0."
            )

        if np.any(self.stiffness <= 0):
            raise ValueError(
                "All stiffness values must be greater than 0."
            )

        if np.any(self.damping < 0):
            raise ValueError(
                "All damping values must be greater than "
                "or equal to 0."
            )


    # ==================================================
    # MASS MATRIX
    # ==================================================

    def build_mass_matrix(self):

        return np.diag(
            self.masses
        )


    # ==================================================
    # STIFFNESS MATRIX
    # ==================================================

    def build_stiffness_matrix(self):

        N = self.dof

        K = np.zeros(
            (N, N),
            dtype=float
        )

        # ------------------------------------------------
        # 1 DOF
        # ------------------------------------------------

        if N == 1:

            K[0, 0] = self.stiffness[0]

            return K

        # ------------------------------------------------
        # First floor
        # ------------------------------------------------

        K[0, 0] = (
            self.stiffness[0]
            + self.stiffness[1]
        )

        K[0, 1] = -self.stiffness[1]

        # ------------------------------------------------
        # Intermediate floors
        # ------------------------------------------------

        for i in range(1, N - 1):

            K[i, i - 1] = -self.stiffness[i]

            K[i, i] = (
                self.stiffness[i]
                + self.stiffness[i + 1]
            )

            K[i, i + 1] = -self.stiffness[i + 1]

        # ------------------------------------------------
        # Top floor
        # ------------------------------------------------

        K[N - 1, N - 2] = -self.stiffness[N - 1]

        K[N - 1, N - 1] = self.stiffness[N - 1]

        return K


    # ==================================================
    # DAMPING MATRIX
    # ==================================================

    def build_damping_matrix(self):

        N = self.dof

        C = np.zeros(
            (N, N),
            dtype=float
        )

        # ------------------------------------------------
        # 1 DOF
        # ------------------------------------------------

        if N == 1:

            C[0, 0] = self.damping[0]

            return C

        # ------------------------------------------------
        # First floor
        # ------------------------------------------------

        C[0, 0] = (
            self.damping[0]
            + self.damping[1]
        )

        C[0, 1] = -self.damping[1]

        # ------------------------------------------------
        # Intermediate floors
        # ------------------------------------------------

        for i in range(1, N - 1):

            C[i, i - 1] = -self.damping[i]

            C[i, i] = (
                self.damping[i]
                + self.damping[i + 1]
            )

            C[i, i + 1] = -self.damping[i + 1]

        # ------------------------------------------------
        # Top floor
        # ------------------------------------------------

        C[N - 1, N - 2] = -self.damping[N - 1]

        C[N - 1, N - 1] = self.damping[N - 1]

        return C


    # ==================================================
    # EXTERNAL FORCE DEFINITION
    # ==================================================

    def set_forces(self, forces):

        if not isinstance(forces, list):
            raise ValueError(
                "Forces must be provided as a list."
            )

        validated_forces = []

        for force in forces:

            # --------------------------------------------
            # Required keys
            # --------------------------------------------

            required_keys = {
                "floor",
                "amplitude",
                "angular_frequency"
            }

            if not required_keys.issubset(force.keys()):
                raise ValueError(
                    "Each force must contain "
                    "'floor', 'amplitude', and "
                    "'angular_frequency'."
                )

            floor = force["floor"]

            amplitude = force["amplitude"]

            angular_frequency = (
                force["angular_frequency"]
            )

            # --------------------------------------------
            # Floor validation
            # --------------------------------------------

            if not isinstance(
                floor,
                (int, np.integer)
            ):
                raise ValueError(
                    "Force floor must be an integer."
                )

            if floor < 0 or floor >= self.dof:
                raise ValueError(
                    f"Floor must be between "
                    f"0 and {self.dof - 1}."
                )

            # --------------------------------------------
            # Amplitude validation
            # --------------------------------------------

            if amplitude < 0:
                raise ValueError(
                    "Force amplitude must be "
                    "non-negative."
                )

            # --------------------------------------------
            # Frequency validation
            # --------------------------------------------

            if angular_frequency <= 0:
                raise ValueError(
                    "Angular frequency must "
                    "be greater than 0."
                )

            # --------------------------------------------
            # Store a copy
            # --------------------------------------------

            validated_forces.append(
                {
                    "floor": floor,
                    "amplitude": float(amplitude),
                    "angular_frequency": float(
                        angular_frequency
                    )
                }
            )

        self.forces = validated_forces


    # ==================================================
    # FORCE AT TIME t
    # ==================================================

    def external_force(self, time):

        if not self.forces:

            raise ValueError(
                "No external forces have been defined. "
                "Call set_forces() before simulation."
            )

        F = np.zeros(
            self.dof,
            dtype=float
        )

        for force in self.forces:

            floor = force["floor"]

            amplitude = force["amplitude"]

            angular_frequency = (
                force["angular_frequency"]
            )

            F[floor] += (
                amplitude
                * np.sin(
                    angular_frequency * time
                )
            )

        return F


    # ==================================================
    # SYSTEM DYNAMICS
    # ==================================================

    def structural_dynamics(
        self,
        time,
        z
    ):

        N = self.dof

        # ------------------------------------------------
        # Convert flattened state into N x 2
        # ------------------------------------------------
        #
        # [x1, v1, x2, v2, ...]
        #
        # becomes
        #
        # [[x1, v1],
        #  [x2, v2],
        #  ...]
        # ------------------------------------------------

        structure = z.reshape(
            N,
            2
        )

        # ------------------------------------------------
        # Displacement
        # ------------------------------------------------

        x = structure[:, 0]

        # ------------------------------------------------
        # Velocity
        # ------------------------------------------------

        v = structure[:, 1]

        # ------------------------------------------------
        # Structural matrices
        # ------------------------------------------------

        M = self.build_mass_matrix()

        C = self.build_damping_matrix()

        K = self.build_stiffness_matrix()

        # ------------------------------------------------
        # External force
        # ------------------------------------------------

        F = self.external_force(time)

        # ------------------------------------------------
        # Equation of motion
        #
        # M a = F - C v - K x
        # ------------------------------------------------

        accelerations = np.linalg.solve(
            M,
            F
            - C @ v
            - K @ x
        )

        # ------------------------------------------------
        # State derivative
        #
        # dz/dt = [v, a]
        # ------------------------------------------------

        dZ = np.column_stack(
            (
                v,
                accelerations
            )
        )

        return dZ.flatten()




# ======================================================
# DAMAGE
# ======================================================
def apply_damage(
    stiffness,
    damage_elements,
    damage_severities
):
    """Reduce the stiffness of specified elements."""

    stiffness = np.asarray(stiffness, dtype=np.float64)
    elements = list(damage_elements)
    severities = [float(s) for s in damage_severities]

    if len(elements) != len(severities):
        raise ValueError(
            "damage_elements and damage_severities "
            "must have the same length."
        )

    if len(set(elements)) != len(elements):
        raise ValueError("Duplicate damage elements are not allowed.")

    damaged_stiffness = stiffness.copy()

    for element, severity in zip(elements, severities):

        if not isinstance(element, (int, np.integer)):
            raise ValueError("Each damage element must be an integer.")

        if not 0 <= element < len(stiffness):
            raise ValueError(
                f"Damage element must be between 0 and {len(stiffness) - 1}."
            )

        if not np.isfinite(severity) or not 0.0 <= severity < 1.0:
            raise ValueError(
                "Damage severity must be finite and in [0, 1)."
            )

        damaged_stiffness[element] *= (1.0 - severity)

    return damaged_stiffness

# ======================================================
# SENSOR SIMULATION
# ======================================================

def simulate_sensors(
    case,
    sensor_dofs
):
    """
    Extract structural acceleration responses
    corresponding to sensor locations.
    """

    # --------------------------------------------------
    # Convert sensor locations
    # --------------------------------------------------

    sensor_dofs = np.asarray(
        sensor_dofs,
        dtype=int
    )

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    if sensor_dofs.ndim != 1:

        raise ValueError(
            "sensor_dofs must be a 1-dimensional list."
        )

    if len(sensor_dofs) == 0:

        raise ValueError(
            "At least one sensor DOF is required."
        )

    if len(np.unique(sensor_dofs)) != len(sensor_dofs):

        raise ValueError(
            "Sensor DOFs must be unique."
        )

    # --------------------------------------------------
    # Get acceleration
    # --------------------------------------------------

    acceleration = np.asarray(
        case["acceleration"],
        dtype=float
    )

    number_of_dofs = acceleration.shape[1]

    # --------------------------------------------------
    # Check sensor locations
    # --------------------------------------------------

    if np.any(sensor_dofs < 0):

        raise ValueError(
            "Sensor DOFs cannot be negative."
        )

    if np.any(sensor_dofs >= number_of_dofs):

        raise ValueError(
            f"Sensor DOFs must be between "
            f"0 and {number_of_dofs - 1}."
        )

    # --------------------------------------------------
    # Extract sparse sensor measurements
    # --------------------------------------------------

    sensor_acceleration = acceleration[
        :,
        sensor_dofs
    ]

    sensor_data = {
        "time": np.asarray(
            case["time"],
            dtype=float
        ),

        "sensor_dofs": sensor_dofs,

        "acceleration": sensor_acceleration
    }

    return sensor_data


# ======================================================
# NOISE
# ======================================================

def add_noise(
    signal,
    noise_level=0.05,
    random_seed=None
):
    """
    Add Gaussian measurement noise.

    The noise standard deviation for each sensor is:

        noise_level × signal standard deviation

    Example:

        noise_level = 0.05

    means the noise standard deviation is 5%
    of the sensor signal standard deviation.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    # --------------------------------------------------
    # Validate noise level
    # --------------------------------------------------

    if noise_level < 0:

        raise ValueError(
            "Noise level must be non-negative."
        )

    # --------------------------------------------------
    # Random generator
    # --------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    # --------------------------------------------------
    # Signal standard deviation
    # --------------------------------------------------

    signal_std = np.std(
        signal,
        axis=0
    )

    # --------------------------------------------------
    # Generate Gaussian noise
    # --------------------------------------------------

    noise = (
        noise_level
        * signal_std
        * rng.standard_normal(
            signal.shape
        )
    )

    # --------------------------------------------------
    # Add noise
    # --------------------------------------------------

    noisy_signal = signal + noise

    return noisy_signal


# ======================================================
# SENSOR NOISE
# ======================================================

def add_sensor_noise(
    sensor_data,
    noise_level=0.05,
    random_seed=None
):
    """
    Add measurement noise to sensor acceleration data.
    """

    # --------------------------------------------------
    # Create a copy
    # --------------------------------------------------

    noisy_sensor_data = sensor_data.copy()

    # --------------------------------------------------
    # Add noise
    # --------------------------------------------------

    noisy_sensor_data["acceleration"] = add_noise(
        sensor_data["acceleration"],
        noise_level=noise_level,
        random_seed=random_seed
    )

    # --------------------------------------------------
    # Store noise information
    # --------------------------------------------------

    noisy_sensor_data["noise_level"] = (
        noise_level
    )

    return noisy_sensor_data


# # ======================================================
# # NUMERICAL SIMULATION
# # ======================================================

# def simulate_structure(structure):

#     N = structure.dof

#     # Initial displacement
#     x0 = np.zeros(N)

#     # Initial velocity
#     v0 = np.zeros(N)

#     # Flatten initial state
#     z0 = np.column_stack(
#         (x0, v0)
#     ).flatten()

#     # Simulation time
#     t_start = 0.0
#     t_end = 10.0

#     t_eval = np.linspace(
#         t_start,
#         t_end,
#         5000
#     )

#     # Solve ODE
#     solution = solve_ivp(
#         fun=structure.structural_dynamics,

#         t_span=(
#             t_start,
#             t_end
#         ),

#         y0=z0,

#         t_eval=t_eval,

#         method="RK45"
#     )

#     if not solution.success:

#         raise RuntimeError(
#             f"Solver failed: "
#             f"{solution.message}"
#         )

#     # ----------------------------------------------
#     # Convert solution back to:
#     #
#     # (time, floor, [x/v])
#     # ----------------------------------------------

#     Z = solution.y.T.reshape(
#         len(solution.t),
#         N,
#         2
#     )

#     # Displacement
#     x = Z[:, :, 0]

#     # Velocity
#     v = Z[:, :, 1]

#     # Acceleration
#     a = np.zeros_like(x)

#     M = structure.build_mass_matrix()

#     C = structure.build_damping_matrix()

#     K = structure.build_stiffness_matrix()

#     # Calculate acceleration
#     # from governing equation
#     for i, time in enumerate(solution.t):

#         F = structure.external_force(time)

#         a[i] = np.linalg.solve(
#             M,
#             F
#             - C @ v[i]
#             - K @ x[i]
#         )

#     return (
#         solution.t,
#         x,
#         v,
#         a
#     )