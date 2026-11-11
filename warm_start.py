"""Compatibility demo for the Hybrid A* warm-start planner.

The reusable implementation now lives in ``MPC/mpc_controller/hybrid_astar.py``
and is invoked by the ROS trajectory-planning node.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from MPC.mpc_controller.hybrid_astar import (
    HybridAStar,
    HybridAStarConfig,
    Pose,
    Rectangle,
)


def save_path_csv(path, filename="planned_path.csv"):
    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["x", "y", "heading_rad", "direction"])
        writer.writerows(
            (point.x, point.y, point.heading, point.direction) for point in path
        )


def main():
    obstacles = (
        Rectangle(2.0, -1.5, 1.0, 4.0),
        Rectangle(4.5, 2.0, 3.0, 1.0),
        Rectangle(4.5, -4.0, 1.0, 3.0),
        Rectangle(-3.5, 3.0, 3.0, 1.0),
        Rectangle(-5.0, -3.5, 2.5, 2.5),
        Rectangle(7.0, 5.0, 1.5, 2.0),
    )
    config = HybridAStarConfig(
        wheelbase=1.0,
        max_steering=np.pi / 3.0,
        speed=1.0,
        edge_length=2.0,
        integration_dt=0.01,
        xy_resolution=0.25,
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        collision_radius=0.0,
    )
    start = Pose(0.0, 0.0, -np.pi / 2.0)
    goal = Pose(9.0, 9.0, 0.0)
    path = HybridAStar(config, obstacles).plan(start, goal)
    if path is None:
        raise RuntimeError("Hybrid A* found no path")
    save_path_csv(path)

    plt.plot([point.x for point in path], [point.y for point in path], label="A* path")
    for obstacle in obstacles:
        plt.gca().add_patch(
            plt.Rectangle(
                (obstacle.x, obstacle.y), obstacle.width, obstacle.height,
                color="tab:red", alpha=0.35,
            )
        )
    plt.scatter([start.x, goal.x], [start.y, goal.y], color=["green", "black"])
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
