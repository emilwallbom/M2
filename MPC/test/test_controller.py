import numpy as np

from mpc_controller.constraints import build_constraints
from mpc_controller.controller import MPCConfig, MPCController
from mpc_controller.model import step


def test_controller_solves_and_respects_dynamics_and_limits():
    config = MPCConfig(horizon=8, dt=0.1, solver_max_cpu_time=1.0)
    controller = MPCController(config)
    state = np.zeros(4)
    reference = np.zeros((config.horizon + 1, 4))
    reference[:, 0] = np.arange(config.horizon + 1) * 0.03
    reference[:, 3] = 0.3

    result = controller.solve(state, reference, np.zeros(2))

    assert result.success, result.status
    assert result.states.shape == (config.horizon + 1, 4)
    assert result.controls.shape == (config.horizon, 2)
    for index in range(config.horizon):
        predicted = step(
            result.states[index], result.controls[index], config.dt, controller.params
        )
        np.testing.assert_allclose(result.states[index + 1], predicted, atol=1e-6)
    constraint_set = build_constraints(
        result.states,
        result.controls,
        np.zeros(2),
        config.dt,
        controller.params,
    )
    assert constraint_set.is_satisfied(1e-6)


def test_second_solve_uses_valid_shifted_warm_start():
    config = MPCConfig(horizon=6, dt=0.1, solver_max_cpu_time=1.0)
    controller = MPCController(config)
    reference = np.zeros((config.horizon + 1, 4))
    reference[:, 0] = np.arange(config.horizon + 1) * 0.02
    reference[:, 3] = 0.2
    first = controller.solve(np.zeros(4), reference, np.zeros(2))
    assert first.success, first.status

    next_state = step(np.zeros(4), first.control, config.dt, controller.params)
    second = controller.solve(next_state, reference, first.control)
    assert second.success, second.status
    np.testing.assert_allclose(second.states[0], next_state, atol=1e-7)
