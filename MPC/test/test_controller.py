import numpy as np

from mpc_controller.constraints import build_constraints
from mpc_controller.controller import OptimizerConfig, TrajectoryOptimizer
from mpc_controller.costs import trajectory_objective
from mpc_controller.model import step
from mpc_controller.reference import build_initial_guess


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
