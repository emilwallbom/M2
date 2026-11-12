import numpy as np

from mpc_controller.constraints import build_constraints
from mpc_controller.controller import OptimizerConfig, TrajectoryOptimizer
from mpc_controller.costs import trajectory_objective
from mpc_controller.model import step
from mpc_controller.reference import TrajectoryGuess, build_initial_guess


def test_optimizer_fixes_endpoints_and_respects_dynamics_and_limits():
    x = np.linspace(0.0, 0.8, 17)
    path = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    guess = build_initial_guess(path, sample_time=0.15)
    start = guess.states[0].copy()
    goal = guess.states[-1].copy()
    optimizer = TrajectoryOptimizer(
        OptimizerConfig(solver_max_cpu_time=2.0, solver_max_iterations=300)
    )

    result = optimizer.solve(start, goal, guess)

    assert result.success, result.status
    np.testing.assert_allclose(result.states[0], start, atol=1e-7)
    np.testing.assert_allclose(result.states[-1], goal, atol=1e-7)
    for index in range(guess.horizon):
        predicted = step(
            result.states[index], result.controls[index], result.sample_time,
            optimizer.params,
        )
        np.testing.assert_allclose(result.states[index + 1], predicted, atol=1e-6)
    constraints = build_constraints(
        result.states, result.controls, np.zeros(2), result.sample_time,
        optimizer.params,
    )
    assert constraints.is_satisfied(1e-6)
    assert np.isclose(
        result.objective,
        trajectory_objective(
            result.controls, result.sample_time, weights=optimizer.weights
        ),
        atol=1e-6,
    )


def test_warm_start_is_not_a_tracked_reference():
    x = np.linspace(0.0, 0.8, 17)
    y = 0.12 * np.sin(np.pi * x / x[-1])
    yaw = np.arctan2(np.gradient(y, x), np.ones_like(x))
    guess = build_initial_guess(np.column_stack((x, y, yaw)), sample_time=0.15)
    result = TrajectoryOptimizer().solve(
        np.array([0.0, 0.0, yaw[0], 0.0]),
        np.array([0.8, 0.0, yaw[-1], 0.0]),
        guess,
    )
    assert result.success, result.status
    change = np.max(np.linalg.norm(result.states[:, :2] - guess.states[:, :2], axis=1))
    assert change > 0.02


def test_map_bounds_keep_the_whole_footprint_inside():
    import pytest
    from mpc_controller.controller import OptimizerConfig
    from mpc_controller.model import VehicleParams, rotation, translation

    params = VehicleParams()
    n = 12
    states = np.zeros((n + 1, 4))
    states[:, 0] = np.linspace(0.0, 1.0, n + 1)
    states[1:-1, 3] = 0.3
    guess = TrajectoryGuess(states=states, controls=np.zeros((n, 2)), sample_time=0.2)
    bounds = (-0.5, 2.0, -0.4, 0.6)
    result = TrajectoryOptimizer(OptimizerConfig(map_bounds=bounds), params).solve(
        guess.states[0], guess.states[-1], guess
    )
    assert result.success, result.status
    half = np.array([params.length / 2, params.width / 2])
    for state in result.states:
        corners = np.array([
            translation(state, params) + rotation(state) @ (half * [sx, sy])
            for sx in (1, -1) for sy in (1, -1)
        ])
        assert corners[:, 0].min() >= bounds[0] - 1e-6
        assert corners[:, 0].max() <= bounds[1] + 1e-6
        assert corners[:, 1].min() >= bounds[2] - 1e-6
        assert corners[:, 1].max() <= bounds[3] + 1e-6

    # Bounds that cut through the goal footprint must make the problem infeasible.
    too_tight = TrajectoryOptimizer(OptimizerConfig(map_bounds=(-0.5, 1.1, -0.4, 0.6)), params)
    assert not too_tight.solve(guess.states[0], guess.states[-1], guess).success
    with pytest.raises(ValueError):
        OptimizerConfig(map_bounds=(1.0, 0.0, 0.0, 1.0))
