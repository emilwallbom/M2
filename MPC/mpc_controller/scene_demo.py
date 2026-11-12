"""Hybrid A* + OBCA on parking/driving scenes, plotted together.

Scenes (metres): ``bay`` (reverse into a 1.0 m bay), ``parallel`` (parallel
park), ``slalom`` (forward only), ``uturn`` (turn around in a corridor),
``corner`` (tight L-shaped corridor), and ``paper_tjunction`` (the paper's
T-junction maneuver at Cascar scale).
Run from MPC/:  PYTHONPATH=. python -m mpc_controller.scene_demo --scene parallel
Plots go to MPC/test/ (scene_demo.png for ``bay``, scene_<name>.png otherwise).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .controller import OptimizerConfig, TrajectoryOptimizer
from .hybrid_astar import HybridAStar, HybridAStarConfig, Pose, Rectangle, path_array
from .model import VehicleParams
from .obstacles import (
    axis_aligned_rectangle,
    polytope_vertices,
    trajectory_clearance,
    vehicle_vertices,
)
from .reference import build_initial_guess

TEST_DIR = Path(__file__).resolve().parents[1] / "test"


@dataclass(frozen=True)
class Scene:
    name: str
    map_bounds: tuple  # x_min, x_max, y_min, y_max
    rectangles: tuple  # Rectangle(lower-left x, y, width, height); both planners use these
    start: Pose
    goal: Pose  # rear-axle pose
    collision_radius: float = 0.3


SCENES = {
    # Reverse into a 1.0 m bay between two parked cars, crate in the lane.
    "bay": Scene(
        "bay", (-1.5, 4.5, -0.9, 2.2),
        (Rectangle(-1.5, -0.9, 2.4, 1.4), Rectangle(1.9, -0.9, 2.6, 1.4),
         Rectangle(2.0, 1.0, 0.4, 0.5)),
        Pose(-1.0, 1.2, 0.0), Pose(1.4, -0.3, np.pi / 2),
    ),
    # Parallel park into a 1.8 m long, 0.5 m deep spot (heading stays +x).
    "parallel": Scene(
        "parallel", (-1.5, 4.5, -0.5, 2.0),
        (Rectangle(-1.5, -0.5, 1.7, 1.0), Rectangle(2.0, -0.5, 2.5, 1.0),
         Rectangle(0.2, -0.5, 1.8, 0.5)),
        Pose(-1.2, 1.3, 0.0), Pose(1.0, 0.36, 0.0),
    ),
    # Turn around in a 2.0 m wide dead-end corridor (needs several cusps).
    "uturn": Scene(
        "uturn", (-3.0, 1.0, -1.0, 1.0),
        (Rectangle(0.3, -0.2, 0.3, 0.4),),
        Pose(-1.8, -0.5, 0.0), Pose(-1.8, 0.5, np.pi),
    ),
    # Tight 90-degree corner in a 1.2 m wide L-shaped corridor, going forward.
    "corner": Scene(
        "corner", (-1.0, 5.0, -1.0, 5.0),
        (Rectangle(-1.0, -1.0, 6.0, 1.0), Rectangle(-1.0, 1.2, 3.4, 3.8),
         Rectangle(3.6, -1.0, 1.4, 6.0)),
        Pose(-0.5, 0.6, 0.0), Pose(3.0, 4.4, np.pi / 2),
    ),
    # Forward-only slalom through three walls: no direction change needed.
    "slalom": Scene(
        "slalom", (-1.0, 6.0, -1.5, 1.5),
        (Rectangle(1.0, -1.5, 0.3, 1.8), Rectangle(2.6, -0.3, 0.3, 1.8),
         Rectangle(4.2, -1.5, 0.3, 1.8)),
        Pose(-0.5, 0.0, 0.0), Pose(5.5, 0.0, 0.0),
    ),
    # Cascar-scale recreation of the paper's constrained T-junction maneuver.
    # The tight upper chamber forces a forward/reverse turn before exiting left.
    "paper_tjunction": Scene(
        "paper_tjunction", (-3.0, 3.0, 0.0, 1.7),
        (Rectangle(-3.0, 0.0, 2.675, 1.0),
         Rectangle(0.325, 0.0, 2.675, 1.0),
         Rectangle(-3.0, 1.5, 6.0, 0.2)),
        Pose(0.0, 0.35, np.pi / 2), Pose(-1.5, 1.25, np.pi),
        0.14,
    ),
}


def default_output(name: str) -> Path:
    return TEST_DIR / ("scene_demo.png" if name == "bay" else f"scene_{name}.png")


def run_scene(scene: Scene = SCENES["bay"], *, collision_radius=None, sample_time=0.15,
              dual_init="optimal_separation", bound_map=True):
    params = VehicleParams()
    b = scene.map_bounds
    astar_config = HybridAStarConfig(
        x_min=b[0], x_max=b[1], y_min=b[2], y_max=b[3],
        collision_radius=(scene.collision_radius if collision_radius is None
                          else collision_radius),
        max_expansions=100_000,
    )
    started = perf_counter()
    path = HybridAStar(astar_config, scene.rectangles).plan(scene.start, scene.goal)
    astar_time = perf_counter() - started
    if path is None:
        raise RuntimeError(f"Hybrid A* found no path in scene {scene.name!r}")
    path = path_array(path)

    guess = build_initial_guess(path, sample_time=sample_time, params=params)
    obstacles = tuple(
        axis_aligned_rectangle(r.x + r.width / 2, r.y + r.height / 2, r.width, r.height)
        for r in scene.rectangles
    )
    optimizer = TrajectoryOptimizer(
        OptimizerConfig(dual_init=dual_init, map_bounds=b if bound_map else None),
        params, obstacles=obstacles,
    )
    start = np.array([scene.start.x, scene.start.y, scene.start.heading, 0.0])
    goal = np.array([scene.goal.x, scene.goal.y, scene.goal.heading, 0.0])
    result = optimizer.solve(start, goal, guess)
    clearance = (
        trajectory_clearance(result.states, obstacles, params) if result.success else None
    )
    return {
        "scene": scene,
        "path": path,
        "guess": guess,
        "result": result,
        "obstacles": obstacles,
        "params": params,
        "clearance": clearance,
        "guess_clearance": trajectory_clearance(guess.states, obstacles, params),
        "astar_time": astar_time,
        "min_clearance": optimizer.config.minimum_clearance,
    }


def plot_scene(scene_run, output: Path, *, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    scene = scene_run["scene"]
    path, result, params = scene_run["path"], scene_run["result"], scene_run["params"]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for index, obstacle in enumerate(scene_run["obstacles"]):
        axis.add_patch(Polygon(polytope_vertices(obstacle), color="0.65",
                               label="obstacles" if index == 0 else None))
    forward = path[:, 3] > 0
    axis.plot(path[:, 0], path[:, 1], "-", color="tab:red", lw=1, alpha=0.5)
    axis.plot(path[forward, 0], path[forward, 1], ".", color="tab:red", ms=3,
              label="Hybrid A* (forward)")
    axis.plot(path[~forward, 0], path[~forward, 1], ".", color="tab:orange", ms=3,
              label="Hybrid A* (reverse)")
    if result.success:
        states = result.states
        for state in states[:: max(1, len(states) // 40)]:
            axis.add_patch(Polygon(vehicle_vertices(state, params), fill=False,
                                   edgecolor="green", alpha=0.55, lw=0.8))
        axis.plot(states[:, 0], states[:, 1], "-", color="tab:blue", lw=2,
                  label="OBCA optimized (rear axle)")
    for pose, name in ((scene.start, "start"), (scene.goal, "goal")):
        axis.annotate(name, (pose.x, pose.y), xytext=(pose.x, pose.y + 0.15),
                      ha="center", fontsize=9)
        axis.arrow(pose.x, pose.y, 0.2 * np.cos(pose.heading),
                   0.2 * np.sin(pose.heading), head_width=0.05, color="k")
    b = scene.map_bounds
    axis.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False,
                                 ls="--", ec="k", lw=0.8, label="map bounds"))
    low, high = np.array([b[0], b[2]]), np.array([b[1], b[3]])
    if result.success:
        low = np.minimum(low, result.states[:, :2].min(axis=0) - 0.4)
        high = np.maximum(high, result.states[:, :2].max(axis=0) + 0.4)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal")
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    if result.success:
        title = (f"[{scene.name}] Hybrid A*: {len(path)} poses, {scene_run['astar_time']:.1f}s  |  "
                 f"OBCA: {result.status}, {result.iterations} it, {result.solve_time:.2f}s, "
                 f"{result.total_time:.1f}s maneuver, min clearance "
                 f"{scene_run['clearance'].min():.3f} m")
    else:
        title = f"[{scene.name}] OBCA FAILED: {result.status}"
    axis.set_title(title, fontsize=9)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="bay", choices=sorted(SCENES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dual-init", default="optimal_separation",
                        choices=("optimal_separation", "legacy"))
    parser.add_argument("--show", action="store_true")
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO)
    run = run_scene(SCENES[options.scene], dual_init=options.dual_init)
    r = run["result"]
    print(f"A*: {len(run['path'])} poses in {run['astar_time']:.1f}s; guess min clearance "
          f"{run['guess_clearance'].min():.3f} m")
    print(f"OBCA success={r.success} status={r.status} iters={r.iterations} "
          f"solve={r.solve_time:.2f}s")
    if r.success:
        print(f"T_opt={r.sample_time:.3f}s total={r.total_time:.1f}s "
              f"min clearance={run['clearance'].min():.3f} m (required {run['min_clearance']})")
    output = options.output or default_output(options.scene)
    plot_scene(run, output, show=options.show)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
