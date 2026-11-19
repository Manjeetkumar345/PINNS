import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from models import (calculate_response_derivatives, calculate_physics_residual)


# ======================================================
# METRIC FUNCTIONS
# ======================================================

def calculate_metrics(
    true,
    predicted
):
    """
    Calculate general prediction metrics.
    """

    true = np.asarray(
        true,
        dtype=np.float64
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64
    )

    error = predicted - true

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            np.square(error)
        )
    )

    max_error = np.max(
        np.abs(error)
    )

    relative_l2 = (
        np.linalg.norm(error)
        /
        (
            np.linalg.norm(true)
            + 1e-12
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Max_Absolute_Error": float(max_error),
        "Relative_L2_Error": float(relative_l2)
    }


# ======================================================
# PER-FLOOR METRICS
# ======================================================

def calculate_per_floor_metrics(
    true_displacement,
    predicted_displacement,
    true_velocity,
    predicted_velocity,
    true_acceleration,
    predicted_acceleration
):
    """
    Calculate metrics separately for every DOF.
    """

    number_of_dofs = (
        true_displacement.shape[1]
    )

    results = []

    for dof in range(number_of_dofs):

        x_metrics = calculate_metrics(
            true_displacement[:, dof],
            predicted_displacement[:, dof]
        )

        v_metrics = calculate_metrics(
            true_velocity[:, dof],
            predicted_velocity[:, dof]
        )

        a_metrics = calculate_metrics(
            true_acceleration[:, dof],
            predicted_acceleration[:, dof]
        )

        results.append({

            "Floor": dof + 1,

            "X_MAE":
                x_metrics["MAE"],

            "X_RMSE":
                x_metrics["RMSE"],

            "X_Relative_L2":
                x_metrics["Relative_L2_Error"],

            "V_MAE":
                v_metrics["MAE"],

            "V_RMSE":
                v_metrics["RMSE"],

            "V_Relative_L2":
                v_metrics["Relative_L2_Error"],

            "A_MAE":
                a_metrics["MAE"],

            "A_RMSE":
                a_metrics["RMSE"],

            "A_Relative_L2":
                a_metrics["Relative_L2_Error"]
        })

    return pd.DataFrame(
        results
    )


# ======================================================
# PREDICT IN BATCHES
# ======================================================

def predict_in_batches(
    model,
    time,
    batch_size=512
):
    """
    Predict x, v and a in batches.

    This avoids calculating second derivatives for
    all 8000 time points simultaneously.
    """

    x_predictions = []
    v_predictions = []
    a_predictions = []

    number_of_points = (
        time.shape[0]
    )

    for start in range(
        0,
        number_of_points,
        batch_size
    ):

        end = min(
            start + batch_size,
            number_of_points
        )

        t_batch = time[
            start:end
        ]

        (
            x_hat,
            v_hat,
            a_hat
        ) = calculate_response_derivatives(
            model,
            t_batch
        )

        x_predictions.append(
            x_hat.numpy()
        )

        v_predictions.append(
            v_hat.numpy()
        )

        a_predictions.append(
            a_hat.numpy()
        )

    return (
        np.concatenate(
            x_predictions,
            axis=0
        ),

        np.concatenate(
            v_predictions,
            axis=0
        ),

        np.concatenate(
            a_predictions,
            axis=0
        )
    )


# ======================================================
# PHYSICS VALIDATION
# ======================================================

def validate_physics(
    model,
    time,
    M,
    C,
    K,
    force_history,
    batch_size=512
):
    """
    Validate the governing structural equation
    over the complete trajectory.
    """

    residual_batches = []

    number_of_points = (
        time.shape[0]
    )

    for start in range(
        0,
        number_of_points,
        batch_size
    ):

        end = min(
            start + batch_size,
            number_of_points
        )

        t_batch = time[
            start:end
        ]

        force_batch = force_history[
            start:end
        ]

        (_,_,_,residual) = calculate_physics_residual(
            model = model,
            t= t_batch,
            M=M,
            C=C,
            K=K,
            force_batch=force_batch
        )

        residual_batches.append(
            residual.numpy()
        )

    residual = np.concatenate(
        residual_batches,
        axis=0
    )

    residual_abs = np.abs(
        residual
    )

    physics_mae = np.mean(
        residual_abs
    )

    physics_rmse = np.sqrt(
        np.mean(
            np.square(residual)
        )
    )

    physics_max = np.max(
        residual_abs
    )

    return {
        "MAE": float(
            physics_mae
        ),

        "RMSE": float(
            physics_rmse
        ),

        "Max_Absolute_Residual": float(
            physics_max
        ),

        "residual": residual
    }


# ======================================================
# INITIAL CONDITION VALIDATION
# ======================================================

def validate_initial_conditions(
    model
):
    """
    Check:

        x(0) = 0
        v(0) = 0
    """

    t0 = tf.zeros(
        (1, 1),
        dtype=tf.float32
    )

    (
        x0_hat,
        v0_hat,
        _
    ) = calculate_response_derivatives(
        model,
        t0
    )

    x0 = x0_hat.numpy()
    v0 = v0_hat.numpy()

    return {

        "displacement_values":
            x0.flatten().tolist(),

        "velocity_values":
            v0.flatten().tolist(),

        "maximum_displacement_error":
            float(
                np.max(
                    np.abs(x0)
                )
            ),

        "maximum_velocity_error":
            float(
                np.max(
                    np.abs(v0)
                )
            )
    }


# ======================================================
# PLOT LOSS HISTORY
# ======================================================

def plot_loss_history(
    history,
    output_dir
):

    epochs = np.arange(
        1,
        len(
            history["total_loss"]
        ) + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        history["total_loss"]
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Total Loss"
    )

    plt.title(
        "Healthy PINN Total Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "total_loss.png"
        )
    )

    plt.close()


    # --------------------------------------------------
    # Data loss
    # --------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        history["data_loss"]
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Data Loss"
    )

    plt.title(
        "Healthy PINN Data Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "data_loss.png"
        )
    )

    plt.close()


    # --------------------------------------------------
    # Physics loss
    # --------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        history["physics_loss"]
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Physics Loss"
    )

    plt.title(
        "Healthy PINN Physics Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "physics_loss.png"
        )
    )

    plt.close()


    # --------------------------------------------------
    # IC loss
    # --------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        history["ic_loss"]
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Initial Condition Loss"
    )

    plt.title(
        "Healthy PINN Initial Condition Loss"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "ic_loss.png"
        )
    )

    plt.close()


# ======================================================
# RESPONSE PLOTS
# ======================================================

def plot_response_comparison(
    time,
    true_data,
    predicted_data,
    quantity_name,
    output_dir
):
    """
    Plot true vs predicted response for every floor.
    """

    number_of_dofs = (
        true_data.shape[1]
    )

    for dof in range(
        number_of_dofs
    ):

        plt.figure()

        plt.plot(
            time,
            true_data[:, dof],
            label="True"
        )

        plt.plot(
            time,
            predicted_data[:, dof],
            label="PINN"
        )

        plt.xlabel(
            "Time"
        )

        plt.ylabel(
            quantity_name
        )

        plt.title(
            f"{quantity_name} - Floor {dof + 1}"
        )

        plt.legend()

        plt.grid(
            True
        )

        plt.tight_layout()

        filename = (
            f"{quantity_name.lower()}_"
            f"floor_{dof + 1}.png"
        )

        plt.savefig(
            os.path.join(
                output_dir,
                filename
            )
        )

        plt.close()


# ======================================================
# COMPLETE HEALTHY PINN VALIDATION
# ======================================================

def validate_healthy_pinn(
    model,
    training_data,
    history,
    output_dir="results/healthy_pinn",
    batch_size=512
):
    """
    Complete healthy PINN validation.

    Validation includes:

        1. Displacement
        2. Velocity
        3. Acceleration
        4. Physics residual
        5. Initial conditions
        6. Per-floor metrics
        7. Training history
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------
    # Extract data
    # --------------------------------------------------

    t = training_data[
        "time"
    ]

    x_true = training_data[
        "displacement"
    ]

    v_true = training_data[
        "velocity"
    ]

    a_true = training_data[
        "acceleration"
    ]

    M = training_data[
        "mass_matrix"
    ]

    C = training_data[
        "damping_matrix"
    ]

    K = training_data[
        "stiffness_matrix"
    ]

    force_history = training_data[
        "force_history"
    ]

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    print(
        "\n========== PINN PREDICTION =========="
    )

    (
        x_hat,
        v_hat,
        a_hat
    ) = predict_in_batches(
        model=model,
        time=t,
        batch_size=batch_size
    )

    print(
        "Prediction complete."
    )

    # --------------------------------------------------
    # Convert true data to NumPy
    # --------------------------------------------------

    x_true = x_true.numpy()

    v_true = v_true.numpy()

    a_true = a_true.numpy()

    force_history = force_history.numpy()

    M = M.numpy()

    C = C.numpy()

    K = K.numpy()

    time = t.numpy().reshape(-1)

    # --------------------------------------------------
    # Global response metrics
    # --------------------------------------------------

    displacement_metrics = calculate_metrics(
        x_true,
        x_hat
    )

    velocity_metrics = calculate_metrics(
        v_true,
        v_hat
    )

    acceleration_metrics = calculate_metrics(
        a_true,
        a_hat
    )

    # --------------------------------------------------
    # Physics residual
    # --------------------------------------------------

    print(
        "\n========== PHYSICS VALIDATION =========="
    )

    physics_results = validate_physics(
        model=model,
        time=t,
        M=tf.convert_to_tensor(
            M,
            dtype=tf.float32
        ),
        C=tf.convert_to_tensor(
            C,
            dtype=tf.float32
        ),
        K=tf.convert_to_tensor(
            K,
            dtype=tf.float32
        ),
        force_history=tf.convert_to_tensor(
            force_history,
            dtype=tf.float32
        ),
        batch_size=batch_size
    )

    physics_residual = physics_results[
        "residual"
    ]

    # Remove large residual array from JSON later

    physics_metrics = {
        "MAE":
            physics_results["MAE"],

        "RMSE":
            physics_results["RMSE"],

        "Max_Absolute_Residual":
            physics_results[
                "Max_Absolute_Residual"
            ]
    }

    # --------------------------------------------------
    # Initial conditions
    # --------------------------------------------------

    print(
        "\n========== INITIAL CONDITIONS =========="
    )

    ic_metrics = (
        validate_initial_conditions(
            model
        )
    )

    # --------------------------------------------------
    # Per-floor metrics
    # --------------------------------------------------

    floor_metrics = (
        calculate_per_floor_metrics(
            true_displacement=x_true,
            predicted_displacement=x_hat,
            true_velocity=v_true,
            predicted_velocity=v_hat,
            true_acceleration=a_true,
            predicted_acceleration=a_hat
        )
    )

    # --------------------------------------------------
    # Save floor metrics
    # --------------------------------------------------

    floor_metrics.to_csv(
        os.path.join(
            output_dir,
            "per_floor_metrics.csv"
        ),
        index=False
    )

    # --------------------------------------------------
    # Plot losses
    # --------------------------------------------------

    plot_loss_history(
        history=history,
        output_dir=output_dir
    )

    # --------------------------------------------------
    # Plot response comparisons
    # --------------------------------------------------

    plot_response_comparison(
        time=time,
        true_data=x_true,
        predicted_data=x_hat,
        quantity_name="Displacement",
        output_dir=output_dir
    )

    plot_response_comparison(
        time=time,
        true_data=v_true,
        predicted_data=v_hat,
        quantity_name="Velocity",
        output_dir=output_dir
    )

    plot_response_comparison(
        time=time,
        true_data=a_true,
        predicted_data=a_hat,
        quantity_name="Acceleration",
        output_dir=output_dir
    )

    # --------------------------------------------------
    # Final metrics dictionary
    # --------------------------------------------------

    metrics = {

        "displacement": displacement_metrics,

        "velocity": velocity_metrics,

        "acceleration": acceleration_metrics,

        "physics": physics_metrics,

        "initial_conditions": {

            "maximum_displacement_error":
                ic_metrics[
                    "maximum_displacement_error"
                ],

            "maximum_velocity_error":
                ic_metrics[
                    "maximum_velocity_error"
                ]
        }
    }

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    with open(
        os.path.join(
            output_dir,
            "metrics.json"
        ),
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "\n========== GLOBAL RESPONSE METRICS =========="
    )

    print(
        "\nDisplacement:"
    )

    for key, value in (
        displacement_metrics.items()
    ):

        print(
            f"{key:25s}: {value:.8e}"
        )

    print(
        "\nVelocity:"
    )

    for key, value in (
        velocity_metrics.items()
    ):

        print(
            f"{key:25s}: {value:.8e}"
        )

    print(
        "\nAcceleration:"
    )

    for key, value in (
        acceleration_metrics.items()
    ):

        print(
            f"{key:25s}: {value:.8e}"
        )

    print(
        "\n========== PHYSICS METRICS =========="
    )

    for key, value in (
        physics_metrics.items()
    ):

        print(
            f"{key:25s}: {value:.8e}"
        )

    print(
        "\n========== INITIAL CONDITION METRICS =========="
    )

    print(
        "Maximum displacement IC error:",
        f"{ic_metrics['maximum_displacement_error']:.8e}"
    )

    print(
        "Maximum velocity IC error:",
        f"{ic_metrics['maximum_velocity_error']:.8e}"
    )

    print(
        "\n========== PER-FLOOR METRICS =========="
    )

    print(
        floor_metrics.to_string(
            index=False
        )
    )

    print(
        "\nValidation results saved to:",
        output_dir
    )

    return {
        "metrics":
            metrics,

        "per_floor_metrics":
            floor_metrics,

        "predicted_displacement":
            x_hat,

        "predicted_velocity":
            v_hat,

        "predicted_acceleration":
            a_hat,

        "physics_residual":
            physics_residual
    }


