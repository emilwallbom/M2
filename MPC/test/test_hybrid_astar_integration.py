import numpy as np
from mpc_controller.controller import TrajectoryOptimizer
from mpc_controller.hybrid_astar import (
    HybridAStar,
    HybridAStarConfig,
    Pose,
    Rectangle,
    path_array,
)
from mpc_controller.obstacles import axis_aligned_rectangle, trajectory_clearance
from mpc_controller.reference import build_initial_guess


def _small_config(**overrides):
    values = {
        "collision_radius": 0.0,
        "x_min": -2.0,
        "x_max": 2.0,
        "y_min": -1.0,
        "y_max": 1.0,
        "max_expansions": 5_000,
    }
    values.update(overrides)
    return HybridAStarConfig(**values)


def test_hybrid_astar_path_warm_starts_obca():
    path = HybridAStar(_small_config()).plan(Pose(0.0, 0.0, 0.0), Pose(1.0, 0.0, 0.0))
    assert path is not None
    guess = build_initial_guess(path_array(path), sample_time=0.15)
    assert guess.horizon < len(path) - 1  # dense search integration was thinned

    result = TrajectoryOptimizer().solve(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        guess,
    )
    assert result.success, result.status
    np.testing.assert_allclose(result.states[0], [0.0, 0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result.states[-1], [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_reverse_motion_is_preserved_in_warm_start():
    path = HybridAStar(_small_config()).plan(
        Pose(0.0, 0.0, 0.0), Pose(-0.7, 0.0, 0.0)
    )
    assert path is not None
    assert any(point.direction == -1 for point in path)
    guess = build_initial_guess(path_array(path), sample_time=0.15)
    assert np.any(guess.states[:, 3] < 0.0)


def test_astar_and_obca_use_the_same_obstacle():
    rectangle = Rectangle(0.5, -0.1, 0.2, 0.2)
    path = HybridAStar(
        _small_config(collision_radius=0.22, x_max=1.7, max_expansions=20_000),
        (rectangle,),
    ).plan(Pose(0.0, 0.0, 0.0), Pose(1.2, 0.0, 0.0))
    assert path is not None

    obstacle = axis_aligned_rectangle(0.6, 0.0, 0.2, 0.2)
    optimizer = TrajectoryOptimizer(obstacles=(obstacle,))
    result = optimizer.solve(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.2, 0.0, 0.0, 0.0]),
        build_initial_guess(path_array(path), sample_time=0.15),
    )
    assert result.success, result.status
    clearance = trajectory_clearance(result.states, (obstacle,), optimizer.params)
    assert np.min(clearance) >= optimizer.config.minimum_clearance - 1e-5
