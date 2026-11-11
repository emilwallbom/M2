import numpy as np

from mpc_controller.reference import build_reference


def test_forward_reference_reaches_and_pads_goal():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = build_reference(
        np.zeros(4), path, horizon=4, dt=1.0, target_speed=0.5
    )

    np.testing.assert_allclose(result.states[:, 0], [0.0, 0.5, 1.0, 1.0, 1.0])
    assert result.states[-1, 3] == 0.0


def test_reference_stops_before_direction_change():
    path = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, -1.0],
        ]
    )
    result = build_reference(
        np.zeros(4), path, horizon=7, dt=1.0, target_speed=0.5
    )

    stop = np.flatnonzero(np.isclose(result.states[:, 0], 2.0))
    assert stop.size > 0
    assert np.all(result.states[stop, 3] == 0.0)
    assert not np.any(result.states[:, 3] < 0.0)

    switched = build_reference(
        np.array([2.0, 0.0, 0.0, 0.0]),
        path,
        horizon=4,
        dt=1.0,
        progress_index=1,
        target_speed=0.5,
    )
    assert np.any(switched.states[1:, 3] < 0.0)


def test_goal_reached_uses_position_and_speed():
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    stopped = build_reference(np.array([1.01, 0.0, 0.0, 0.0]), path, 2, 0.1)
    moving = build_reference(np.array([1.01, 0.0, 0.0, 0.2]), path, 2, 0.1)
    assert stopped.goal_reached
    assert not moving.goal_reached
