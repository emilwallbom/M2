import numpy as np

from mpc_controller.controller import MPCConfig
from mpc_controller.demo import plot_result, run_closed_loop
from mpc_controller.model import IDELTA, VehicleParams


def test_closed_loop_reaches_goal_and_writes_plot(tmp_path):
    config = MPCConfig(solver_max_cpu_time=1.0)
    params = VehicleParams()
    result = run_closed_loop(config=config, params=params)

    assert result.goal_reached
    assert np.linalg.norm(result.states[-1, :2] - result.path[-1, :2]) < 0.05
    assert np.max(np.abs(result.states[:, 3])) <= params.max_speed + 1e-6
    steering_rate = np.diff(
        np.concatenate(([0.0], result.controls[:, IDELTA]))
    ) / config.dt
    assert np.max(np.abs(steering_rate)) <= params.max_steering_rate + 1e-6

    output = tmp_path / "mpc_solution.png"
    plot_result(result, output)
    assert output.exists()
    assert output.stat().st_size > 10_000
