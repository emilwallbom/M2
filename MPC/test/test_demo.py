import numpy as np

from mpc_controller.demo import plot_result, run_optimization


def test_one_shot_demo_optimizes_a_different_safe_trajectory(tmp_path):
    demo = run_optimization()
    optimized = demo.optimized
    assert optimized.success
    assert np.min(demo.clearances) >= demo.required_clearance - 1e-5
    np.testing.assert_allclose(optimized.states[0], demo.warm_start.states[0], atol=1e-7)
    np.testing.assert_allclose(optimized.states[-1], demo.warm_start.states[-1], atol=1e-7)
    displacement = np.max(np.linalg.norm(
        optimized.states[:, :2] - demo.warm_start.states[:, :2], axis=1
    ))
    assert displacement > 0.05
    output = tmp_path / "mpc_solution.png"
    plot_result(demo, output)
    assert output.stat().st_size > 10_000
