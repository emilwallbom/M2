import numpy as np

from mpc_controller.reference import TrajectoryGuess, build_initial_guess


def test_hybrid_path_becomes_initial_values_with_inferred_inputs():
    path = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 1.0], [0.2, 0.02, 0.1, 1.0]]
    )
    guess = build_initial_guess(path, sample_time=0.2)
    assert isinstance(guess, TrajectoryGuess)
    assert guess.states.shape == (3, 4)
    assert guess.controls.shape == (2, 2)
    np.testing.assert_allclose(guess.states[:, :3], path[:, :3])
    assert guess.states[0, 3] == 0.0
    assert guess.states[-1, 3] == 0.0
    assert guess.controls[1, 0] > 0.0


def test_reverse_direction_produces_negative_velocity_guess():
    path = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.8, 0.0, 0.0, -1.0], [0.6, 0.0, 0.0, -1.0]]
    )
    guess = build_initial_guess(path, sample_time=0.5)
    assert guess.states[1, 3] < 0.0
