import numpy as np


def simulate_direction():

    # Simulate an incoming RF direction
    true_direction = np.random.uniform(
        0,
        180
    )

    # Add small measurement error
    noise = np.random.normal(
        0,
        3
    )

    estimated_direction = (
        true_direction + noise
    )

    # Keep within 0–180 degrees
    estimated_direction = np.clip(
        estimated_direction,
        0,
        180
    )

    return estimated_direction


if __name__ == "__main__":

    direction = simulate_direction()

    print(
        "Estimated Direction:",
        round(direction, 2),
        "degrees"
    )