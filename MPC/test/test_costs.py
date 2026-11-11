import numpy as np

from mpc_controller.costs import ObjectiveWeights, trajectory_objective


def test_equation_20_cost_components():
    controls = np.array([[0.1, 0.2], [0.2, 0.0]])
    weights = ObjectiveWeights(
        time=2.0,
        steering=3.0,
        acceleration=4.0,
        steering_rate=5.0,
        acceleration_rate=6.0,
    )
    T = 0.5
    rates = np.diff(np.vstack((np.zeros(2), controls)), axis=0) / T
    expected = (
        weights.time * len(controls) * T
        + np.sum(weights.steering * controls[:, 0] ** 2)
        + np.sum(weights.acceleration * controls[:, 1] ** 2)
        + np.sum(weights.steering_rate * rates[:, 0] ** 2)
        + np.sum(weights.acceleration_rate * rates[:, 1] ** 2)
    )
    assert np.isclose(
        trajectory_objective(controls, T, weights=weights), expected
    )


def test_zero_inputs_still_pay_maneuver_time():
    controls = np.zeros((4, 2))
    assert np.isclose(trajectory_objective(controls, 0.2), 0.8)
