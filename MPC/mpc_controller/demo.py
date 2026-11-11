"""Standalone visualization of the paper's OBCA trajectory optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .controller import OptimizerConfig, TrajectoryOptimizer, TrajectoryResult
from .model import IA, IDELTA, IV, VehicleParams
from .obstacles import (
    ConvexPolytope,
    axis_aligned_rectangle,
    polytope_vertices,
    trajectory_clearance,
    vehicle_vertices,
)
from .reference import TrajectoryGuess, build_initial_guess


@dataclass(frozen=True)
class DemoResult:
    warm_start: TrajectoryGuess
    optimized: TrajectoryResult
    obstacles: tuple[ConvexPolytope, ...]
    clearances: np.ndarray
    required_clearance: float
    params: VehicleParams


def make_hybrid_astar_standin(samples: int = 41) -> np.ndarray:
    """Create a coarse collision-free path without implementing Hybrid A*."""
    x = np.linspace(0.0, 3.0, samples)
    height = 0.48
    y = height * np.sin(np.pi * x / 3.0) ** 2
    dydx = height * (np.pi / 3.0) * np.sin(2.0 * np.pi * x / 3.0)
    yaw = np.arctan2(dydx, np.ones_like(dydx))
    return np.column_stack((x, y, yaw, np.ones_like(x)))


def make_demo_obstacles() -> tuple[ConvexPolytope, ...]:
    return (
        axis_aligned_rectangle(
            1.5, 0.0, width=0.38, height=0.30, label="parked object"
        ),
    )


def run_optimization(
    *,
    config: OptimizerConfig | None = None,
    params: VehicleParams | None = None,
) -> DemoResult:
    """Solve once: the A* stand-in is used only as the IPOPT initial value."""
    config = config or OptimizerConfig()
    params = params or VehicleParams()
    path = make_hybrid_astar_standin()
    warm_start = build_initial_guess(path, sample_time=0.15, params=params)
    obstacles = make_demo_obstacles()
    optimizer = TrajectoryOptimizer(config, params, obstacles=obstacles)
    start = warm_start.states[0].copy()
    goal = warm_start.states[-1].copy()
    result = optimizer.solve(start, goal, warm_start)
    if not result.success:
        raise RuntimeError(f"trajectory optimization failed: {result.status}")
    return DemoResult(
        warm_start,
        result,
        obstacles,
        trajectory_clearance(result.states, obstacles, params),
        config.minimum_clearance,
        params,
    )


def _draw_environment(axis, result: DemoResult):
    for index, obstacle in enumerate(result.obstacles):
        vertices = polytope_vertices(obstacle)
        closed = np.vstack((vertices, vertices[0]))
        axis.fill(
            closed[:, 0], closed[:, 1], color="tab:red", alpha=0.35,
            label="Obstacle" if index == 0 else None,
        )
    indices = np.linspace(0, len(result.optimized.states) - 1, 11, dtype=int)
    for footprint_index, state_index in enumerate(np.unique(indices)):
        vertices = vehicle_vertices(result.optimized.states[state_index], result.params)
        closed = np.vstack((vertices, vertices[0]))
        axis.plot(
            closed[:, 0], closed[:, 1], color="tab:blue", alpha=0.32,
            label="Optimized vehicle footprints" if footprint_index == 0 else None,
        )


def plot_result(result: DemoResult, output: Path, *, show: bool = False):
    """Plot the initial guess beside the independently optimized trajectory."""
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    optimized = result.optimized
    state_time = np.arange(len(optimized.states)) * optimized.sample_time
    control_time = np.arange(len(optimized.controls)) * optimized.sample_time

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    path_axis = figure.add_subplot(grid[0, :2])
    clearance_axis = figure.add_subplot(grid[0, 2])
    speed_axis = figure.add_subplot(grid[1, 0])
    input_axis = figure.add_subplot(grid[1, 1])
    cost_axis = figure.add_subplot(grid[1, 2])

    _draw_environment(path_axis, result)
    path_axis.plot(
        result.warm_start.states[:, 0], result.warm_start.states[:, 1],
        "k--", linewidth=1.5, label="Hybrid A* warm start",
    )
    path_axis.plot(
        optimized.states[:, 0], optimized.states[:, 1],
        color="tab:blue", linewidth=2.5, label="OBCA optimized trajectory",
    )
    path_axis.scatter(*optimized.states[0, :2], color="tab:green", s=60, label="Fixed start")
    path_axis.scatter(*optimized.states[-1, :2], color="tab:red", marker="x", s=70, label="Fixed goal")
    path_axis.set_title("A* initializes IPOPT; it is not a tracking reference")
    path_axis.set_xlabel("X [m]")
    path_axis.set_ylabel("Y [m]")
    path_axis.axis("equal")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend(fontsize=8, ncol=2)

    clearance_axis.plot(state_time, result.clearances, color="tab:green")
    clearance_axis.axhline(result.required_clearance, color="tab:red", linestyle="--", label="Required")
    clearance_axis.set_title("Independent body clearance")
    clearance_axis.set_xlabel("Time [s]")
    clearance_axis.set_ylabel("Distance [m]")
    clearance_axis.grid(True, alpha=0.3)
    clearance_axis.legend()

    speed_axis.plot(state_time, optimized.states[:, IV], color="tab:blue")
    speed_axis.set_title("Optimized velocity profile")
    speed_axis.set_xlabel("Time [s]")
    speed_axis.set_ylabel("v [m/s]")
    speed_axis.grid(True, alpha=0.3)

    input_axis.step(control_time, np.rad2deg(optimized.controls[:, IDELTA]), where="post", label="Steering [deg]")
    input_axis.step(control_time, optimized.controls[:, IA], where="post", label="Acceleration [m/s²]")
    input_axis.set_title("Optimized inputs")
    input_axis.set_xlabel("Time [s]")
    input_axis.grid(True, alpha=0.3)
    input_axis.legend()

    labels = ["time", "input", "input rate"]
    values = [optimized.time_cost, optimized.input_cost, optimized.input_rate_cost]
    cost_axis.bar(labels, values, color=["tab:purple", "tab:orange", "tab:cyan"])
    cost_axis.set_title("Equation (20) objective terms")
    cost_axis.set_ylabel("Weighted cost")
    cost_axis.grid(True, axis="y", alpha=0.3)

    displacement = np.max(np.linalg.norm(
        optimized.states[:, :2] - result.warm_start.states[:, :2], axis=1
    ))
    figure.suptitle(
        f"OBCA trajectory optimization | Topt={optimized.sample_time:.3f} s | "
        f"duration={optimized.total_time:.2f} s | max change from warm start={displacement:.3f} m | "
        f"solve={optimized.solve_time:.2f} s"
    )
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("MPC/results/mpc_solution.png"))
    parser.add_argument("--show", action="store_true")
    parsed = parser.parse_args(args)
    result = run_optimization()
    plot_result(result, parsed.output, show=parsed.show)
    displacement = np.max(np.linalg.norm(
        result.optimized.states[:, :2] - result.warm_start.states[:, :2], axis=1
    ))
    print(f"plot: {parsed.output.resolve()}")
    print(f"solver: {result.optimized.status} in {result.optimized.solve_time:.3f} s")
    print(f"Topt / total time: {result.optimized.sample_time:.4f} / {result.optimized.total_time:.3f} s")
    print(f"maximum change from A* warm start: {displacement:.4f} m")
    print(f"minimum body clearance: {np.min(result.clearances):.4f} m")
    print(f"objective: {result.optimized.objective:.4f}")


if __name__ == "__main__":
    main()
