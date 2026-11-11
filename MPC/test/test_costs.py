import numpy as np

from mpc_controller.costs import angle_error, tracking_cost


def test_matching_zero_trajectory_has_zero_cost():
    states = np.zeros((4, 4))
    controls = np.zeros((3, 2))
    assert tracking_cost(states, controls, states, np.zeros(2), dt=0.1) == 0.0


def test_angle_error_wraps_at_pi():
    error = angle_error(-np.pi + 0.1, np.pi - 0.1)
    assert np.isclose(error, 0.2)


def test_terminal_error_is_penalized_more_than_stage_error():
    reference = np.zeros((3, 4))
    controls = np.zeros((2, 2))
    stage_error = reference.copy()
    terminal_error = reference.copy()
    stage_error[0, 0] = 1.0
    terminal_error[-1, 0] = 1.0
    assert tracking_cost(terminal_error, controls, reference) > tracking_cost(
        stage_error, controls, reference
    )
