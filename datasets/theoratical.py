import numpy as np
import pandas as pd
import json
import os

from structure import (
    StructuralDynamics,
    apply_damage,
    simulate_sensors,
    add_sensor_noise,masses,stiffness,damping,N,damage_elements,damage_severities,forces

)

from simulation import (
    simulate_structure
)


DIRECTORY = "./data/theoratical"
HEALTHY = "healthy.json"
DAMAGED = "damaged.json"

file_path1 = os.path.join(DIRECTORY,HEALTHY)
file_path2 = os.path.join(DIRECTORY,DAMAGED)

def convert_to_json_serializable(data):

    if isinstance(data, np.ndarray):
        return data.tolist()

    elif isinstance(data, np.generic):
        return data.item()

    elif isinstance(data, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in data.items()
        }

    elif isinstance(data, list):
        return [
            convert_to_json_serializable(value)
            for value in data
        ]

    return data

# ======================================================
# GENERATE ONE STRUCTURAL CASE
# ======================================================
def generate_case(
    masses,
    stiffness,
    damping,
    damage_elements,
    damage_severities,
    forces
):
    """
    Generate one structural case.

    damage_elements and damage_severities
    define the damage scenario.
    """

    # --------------------------------------------------
    # Apply damage to stiffness
    # --------------------------------------------------

    damaged_stiffness = apply_damage(
        stiffness=stiffness,
        damage_elements=damage_elements,
        damage_severities=damage_severities
    )

    # --------------------------------------------------
    # Create structure
    # --------------------------------------------------

    structure = StructuralDynamics(
        masses=masses,
        stiffness=damaged_stiffness,
        damping=damping
    )

    # --------------------------------------------------
    # Define forces
    # --------------------------------------------------

    structure.set_forces(
        forces
    )

    # --------------------------------------------------
    # Simulate
    # --------------------------------------------------

    t, x, v, a, force_history = simulate_structure(
    structure
    )

    # --------------------------------------------------
    # Determine healthy/damaged
    # --------------------------------------------------

    is_healthy = (
        len(damage_elements) == 0
    )

    # --------------------------------------------------
    # Store case
    # --------------------------------------------------

    case = {

        "mass_matrix":
            structure.build_mass_matrix(),

        "damping_matrix":
            structure.build_damping_matrix(),

        "stiffness_matrix":
            structure.build_stiffness_matrix(),

        "stiffness":
            damaged_stiffness,

        "time": t,

        "displacement": x,

        "velocity": v,

        "acceleration":  a,

        "forces":  forces,

        "force_history": force_history,

        "damage_elements":  list(damage_elements),

        "damage_severities": list(damage_severities),

        "is_healthy": is_healthy
    }

    return case

    
# ======================================================
# GENERATE COMPLETE SIMULATION DATASET
# ======================================================


def generate_dataset(
    masses,
    stiffness,
    damping,
    damage_elements,
    damage_severities,
    forces
):
    """
    Generate structural simulation cases.

    damage_elements and damage_severities represent
    corresponding damage pairs.

    Example:

        damage_elements   = [0, 4]
        damage_severities = [0.10, 0.20]

    means:

        element 0 -> 10% damage
        element 4 -> 20% damage
    """

    dataset = []

    # --------------------------------------------------
    # Validate damage configuration
    # --------------------------------------------------

    if len(damage_elements) != len(damage_severities):

        raise ValueError(
            "damage_elements and damage_severities "
            "must have the same length."
        )

    # --------------------------------------------------
    # HEALTHY CASE
    # --------------------------------------------------

    healthy_case = generate_case(
        masses=masses,
        stiffness=stiffness,
        damping=damping,
        damage_elements=[],
        damage_severities=[],
        forces=forces
    )

    dataset.append(
        healthy_case
    )

    with open(file_path1,"w",encoding="utf-8") as f1:
        healthy_json = convert_to_json_serializable(healthy_case)
        json.dump(healthy_json, f1, indent=4)

    # --------------------------------------------------
    # DAMAGED CASE
    # --------------------------------------------------

    if len(damage_elements) > 0:

        case = generate_case(
            masses=masses,
            stiffness=stiffness,
            damping=damping,
            damage_elements=damage_elements,
            damage_severities=damage_severities,
            forces=forces
        )

        dataset.append(
            case
        )

        with open(file_path2,"w",encoding="utf-8") as f2:
            damaged_json = convert_to_json_serializable(case)
            json.dump(damaged_json, f2, indent=4)

    return dataset


# ======================================================
# PREPARE Healthy data  FOR PINN EXPERIMENT
# ======================================================
def prepare_healthy_pinn_data():

    """
    Prepare one healthy structural case
    for PINN pretraining.

    The PINN uses the complete clean structural
    response and known healthy physics.
    """

    with open(file_path1,"r",encoding="utf-8") as f:
        case = json.load(f)

    # --------------------------------------------------
    # Validate that this is a healthy case
    # --------------------------------------------------

    if not case["is_healthy"]:

        raise ValueError(
            "prepare_healthy_pinn_data() requires "
            "a healthy structural case."
        )

    # --------------------------------------------------
    # PINN training data
    # --------------------------------------------------

    pinn_data = {

        # ==============================================
        # TIME
        # ==============================================

        "time":
            case["time"],


        # ==============================================
        # TRUE STRUCTURAL RESPONSE
        #
        # Used as training data in healthy pretraining
        # ==============================================

        "displacement":
            case["displacement"],

        "velocity":
            case["velocity"],

        "acceleration":
            case["acceleration"],


        # ==============================================
        # STRUCTURAL PHYSICS
        # ==============================================

        "mass_matrix":
            case["mass_matrix"],

        "damping_matrix":
            case["damping_matrix"],

        "stiffness_matrix":
            case["stiffness_matrix"],


        # ==============================================
        # EXTERNAL FORCE
        # ==============================================

        "forces":
            case["forces"],

        "force_history":
            case["force_history"]
    }

    return pinn_data


# ======================================================
# PREPARE DAMAGED DATA FOR PINN EXPERIMENT
# ======================================================

def prepare_damaged_pinn_data():
    """
    Prepare one damaged structural case for PINN training
    or damage-identification experiments.

    The damaged case is generated using the same base
    structural parameters and external forces as the
    healthy case.
    """

    with open(file_path2, "r", encoding="utf-8") as f:
        case = json.load(f)

    # --------------------------------------------------
    # Validate that this is a damaged case
    # --------------------------------------------------

    if case["is_healthy"]:
        raise ValueError(
            "prepare_damaged_pinn_data() requires "
            "a damaged structural case."
        )

    # --------------------------------------------------
    # PINN data
    # --------------------------------------------------

    pinn_data = {
        # TIME
        "time": case["time"],

        # STRUCTURAL RESPONSE
        "displacement": case["displacement"],
        "velocity": case["velocity"],
        "acceleration": case["acceleration"],

        # STRUCTURAL PHYSICS
        "mass_matrix": case["mass_matrix"],
        "damping_matrix": case["damping_matrix"],
        "stiffness_matrix": case["stiffness_matrix"],

        # EXTERNAL FORCE
        "forces": case["forces"],
        "force_history": case["force_history"],

        # DAMAGE INFORMATION
        "damage_elements": case["damage_elements"],
        "damage_severities": case["damage_severities"],
    }

    return pinn_data


dataset_case = generate_case(masses,
    stiffness,
    damping,
    damage_elements,
    damage_severities,
    forces
)


dataset = generate_dataset(
    masses,
    stiffness,
    damping,
    damage_elements,
    damage_severities,
    forces
)




def display_case(case):

    print("=" * 60)
    print("STRUCTURAL DYNAMICS CASE")
    print("=" * 60)

    # 1. Structural Matrices
    print("\n[1] MASS MATRIX")
    print(case["mass_matrix"])

    print("\n[2] DAMPING MATRIX")
    print(case["damping_matrix"])

    print("\n[3] STIFFNESS MATRIX")
    print(case["stiffness_matrix"])

    # 2. Structural Parameters
    print("\n[4] STIFFNESS VALUES")
    print(case["stiffness"])

    # 3. Simulation Information
    print("\n[5] SIMULATION INFORMATION")
    print(f"Time points       : {len(case['time'])}")
    print(f"Duration          : {case['time'][-1]:.2f} seconds")
    print(f"Time step         : {case['time'][1] - case['time'][0]:.6f} seconds")
    print(f"Number of DOFs    : {case['displacement'].shape[1]}")

    # 4. Force Information
    print("\n[6] EXTERNAL FORCES")
    for force in case["forces"]:
        print(
            f"Floor: {force['floor']} | "
            f"Amplitude: {force['amplitude']} | "
            f"Frequency: {force['angular_frequency']}"
        )

    # 5. Damage Information
    print("\n[7] DAMAGE INFORMATION")
    print(f"Damage Elements   : {case['damage_elements']}")
    print(f"Damage Severities : {case['damage_severities']}")
    print(f"Healthy Structure : {case['is_healthy']}")

    # 6. Response Data
    print("\n[8] RESPONSE DATA")
    for key in ["displacement", "velocity", "acceleration"]:
        print(f"{key.capitalize():15} : {case[key].shape}")
        print(f"==============={key}===============")
        print(df:= pd.DataFrame(case[key]).head())

    print("=" * 60)



# display_case(dataset[1])
# display_case(dataset[0])


# print(dataset[1])


