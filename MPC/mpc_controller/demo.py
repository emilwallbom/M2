"""Standalone closed-loop MPC demonstration with diagnostic plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .controller import MPCConfig, MPCController
from .model import IA, IDELTA, IV, VehicleParams, step
from .reference import build_reference


@dataclass(frozen=True)
class DemoResult:
    path: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    reference_speeds: np.ndarray
    solve_times: np.ndarray
    predicted_states: tuple[np.ndarray, ...]
    goal_reached: bool
    dt: float


def make_parking_path(spacing: float = 0.04) -> np.ndarray:
    """Create a feasible forward-then-reverse path without Hybrid A*."""
    forward_x = np.arange(0.0, 1.0, spacing)
    forward = np.column_stack(
        (
            forward_x,
            np.zeros_like(forward_x),
            np.zeros_like(forward_x),
            np.ones_like(forward_x),
        )
    )

    reverse_distance = np.arange(0.0, 0.84, spacing)
    radius = 1.0
    reverse = np.column_stack(
        (
            1.0 - radius * np.sin(reverse_distance / radius),
            radius * (1.0 - np.cos(reverse_distance / radius)),
            -reverse_distance / radius,
            -np.ones_like(reverse_distance),
        )
    )
    return np.vstack((forward, reverse))


def run_closed_loop(
    *,
    config: MPCConfig | None = None,
    params: VehicleParams | None = None,
    max_steps: int = 160,
    target_speed: float = 0.35,
) -> DemoResult:
    """Run MPC against a numerical plant, applying only the first input."""
    config = config or MPCConfig()
    params = params or VehicleParams()
    path = make_parking_path()
    controller = MPCController(config, params)
    state = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0])
    previous_control = np.zeros(2)
    progress_index = 0

    states = [state.copy()]
    controls = []
    reference_speeds = []
    solve_times = []
    predicted_states = []
    goal_reached = False

    for iteration in range(max_steps):
        reference = build_reference(
            state,
            path,
            config.horizon,
            config.dt,
            progress_index=progress_index,
            target_speed=target_speed,
        )
        progress_index = reference.progress_index
        if reference.goal_reached:
            goal_reached = True
            break

        result = controller.solve(state, reference, previous_control)
        if not result.success:
            raise RuntimeError(
                f"MPC failed at closed-loop step {iteration}: {result.status}"
            )
        control = result.control
        if iteration % 10 == 0:
            predicted_states.append(result.states.copy())
        reference_speeds.append(reference.states[0, IV])
        solve_times.append(result.solve_time)
        controls.append(control.copy())
        state = np.asarray(step(state, control, config.dt, params), dtype=float)
        states.append(state.copy())
        previous_control = control

    return DemoResult(
        path,
        np.asarray(states),
        np.asarray(controls),
        np.asarray(reference_speeds),
        np.asarray(solve_times),
        tuple(predicted_states),
        goal_reached,
        config.dt,
    )


def plot_result(result: DemoResult, output: Path, *, show: bool = False):
    """Save a four-panel plot intended for controller review and tuning."""
    if not show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_time = np.arange(len(result.states)) * result.dt
    control_time = np.arange(len(result.controls)) * result.dt
    final_error = np.linalg.norm(result.states[-1, :2] - result.path[-1, :2])

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    path_axis, speed_axis, input_axis, timing_axis = axes.flat

    path_axis.plot(
        result.path[:, 0],
        result.path[:, 1],
        "k--",
        label="Hybrid A* stand-in",
    )
    for index, prediction in enumerate(result.predicted_states):
        path_axis.plot(
            prediction[:, 0],
            prediction[:, 1],
            color="tab:cyan",
            alpha=0.25,
            label="MPC predictions" if index == 0 else None,
        )
    path_axis.plot(
        result.states[:, 0],
        result.states[:, 1],
        color="tab:blue",
        linewidth=2,
        label="Actual",
    )
    path_axis.scatter(
        *result.path[0, :2], color="tab:green", s=70, marker="o", label="Start"
    )
    path_axis.scatter(
        *result.path[-1, :2], color="tab:red", s=80, marker="x", label="Goal"
    )
    changes = np.flatnonzero(result.path[1:, 3] != result.path[:-1, 3]) + 1
    if changes.size:
        path_axis.scatter(
            result.path[changes, 0],
            result.path[changes, 1],
            color="tab:orange",
            s=70,
            marker="D",
            label="Direction change",
        )
    path_axis.set_title("Path and receding-horizon predictions")
    path_axis.set_xlabel("X [m]")
    path_axis.set_ylabel("Y [m]")
    path_axis.axis("equal")
    path_axis.grid(True, alpha=0.3)
    path_axis.legend(fontsize=8)

    speed_axis.plot(state_time, result.states[:, IV], label="Actual speed")
    speed_axis.step(
        control_time,
        result.reference_speeds,
        where="post",
        linestyle="--",
        label="Local reference speed",
    )
    speed_axis.axhline(0.0, color="black", linewidth=0.8)
    speed_axis.set_title("Forward/reverse speed")
    speed_axis.set_xlabel("Time [s]")
    speed_axis.set_ylabel("Speed [m/s]")
    speed_axis.grid(True, alpha=0.3)
    speed_axis.legend()

    input_axis.step(
        control_time,
        np.rad2deg(result.controls[:, IDELTA]),
        where="post",
        label="Steering [deg]",
    )
    input_axis.step(
        control_time,
        result.controls[:, IA],
        where="post",
        label="Acceleration [m/s²]",
    )
    input_axis.set_title("Applied MPC inputs")
    input_axis.set_xlabel("Time [s]")
    input_axis.grid(True, alpha=0.3)
    input_axis.legend()

    timing_axis.plot(
        control_time, 1000.0 * result.solve_times, color="tab:purple"
    )
    timing_axis.axhline(
        1000.0 * result.dt,
        color="tab:red",
        linestyle="--",
        label=f"{1000.0 * result.dt:.0f} ms control period",
    )
    timing_axis.set_title("Nonlinear solver time")
    timing_axis.set_xlabel("Time [s]")
    timing_axis.set_ylabel("Solve time [ms]")
    timing_axis.grid(True, alpha=0.3)
    timing_axis.legend()

    mean_ms = 1000.0 * np.mean(result.solve_times)
    max_ms = 1000.0 * np.max(result.solve_times)
    figure.suptitle(
        f"CasCAR MPC demo | goal={result.goal_reached} | "
        f"final error={final_error:.3f} m | "
        f"solve mean/max={mean_ms:.1f}/{max_ms:.1f} ms"
    )
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("MPC/results/mpc_solution.png"),
        help="PNG output path",
    )
    parser.add_argument("--show", action="store_true", help="open the plot window")
    parsed = parser.parse_args(args)
    result = run_closed_loop()
    plot_result(result, parsed.output, show=parsed.show)
    final_error = np.linalg.norm(result.states[-1, :2] - result.path[-1, :2])
    print(f"plot: {parsed.output.resolve()}")
    print(f"goal reached: {result.goal_reached}")
    print(f"closed-loop steps: {len(result.controls)}")
    print(f"final position error: {final_error:.4f} m")
    print(
        "solver mean/max: "
        f"{1000.0 * np.mean(result.solve_times):.2f}/"
        f"{1000.0 * np.max(result.solve_times):.2f} ms"
    )


if __name__ == "__main__":
    main()
