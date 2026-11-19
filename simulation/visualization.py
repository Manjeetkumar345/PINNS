import matplotlib.pyplot as plt
from structure.structural_model import *


def plot_response(t, response, quantity):

    N = response.shape[1]

    plt.figure(figsize=(10, 6))

    for i in range(N):
        plt.plot(
            t,
            response[:, i],
            label=f"DOF {i + 1}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel(quantity)
    plt.title(f"Structural {quantity} Response")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_healthy_vs_damaged(
    t,
    healthy_response,
    damaged_response,
    quantity
):

    N = healthy_response.shape[1]

    for i in range(N):

        plt.figure(figsize=(10, 5))

        plt.plot(
            t,
            healthy_response[:, i],
            label="Healthy"
        )

        plt.plot(
            t,
            damaged_response[:, i],
            label="Damaged"
        )

        plt.xlabel("Time (s)")
        plt.ylabel(quantity)

        plt.title(
            f"{quantity} Response - DOF {i + 1}"
        )

        plt.legend()
        plt.grid(True)
        plt.show()