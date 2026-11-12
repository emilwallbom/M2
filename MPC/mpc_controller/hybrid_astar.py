"""Hybrid A* warm-start planner adapted from the team's ``warm_start.py``.

The search uses bicycle-model motion primitives with discrete steering angles
and forward/reverse motion. Its result is a geometric initialization for OBCA;
the optimizer remains responsible for dynamic feasibility and clearance.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float
    direction: int = 1


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned obstacle represented by lower-left corner and size."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("rectangle dimensions must be positive")


@dataclass(frozen=True)
class HybridAStarConfig:
    wheelbase: float = 0.275
    max_steering: float = 0.35
    steering_samples: int = 9
    speed: float = 0.35
    edge_length: float = 0.5
    integration_dt: float = 0.05
    xy_resolution: float = 0.08
    heading_resolution: float = np.deg2rad(10.0)
    goal_position_tolerance: float = 0.7
    goal_heading_tolerance: float = np.deg2rad(25.0)
    reverse_cost_factor: float = 1.2
    direction_switch_cost: float = 0.0
    max_expansions: int = 50_000
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0
    collision_radius: float = 0.37

    def __post_init__(self):
        positive = (
            self.wheelbase,
            self.max_steering,
            self.speed,
            self.edge_length,
            self.integration_dt,
            self.xy_resolution,
            self.heading_resolution,
            self.goal_position_tolerance,
            self.goal_heading_tolerance,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "Hybrid A* scale parameters must be finite and positive"
            )
        if self.steering_samples < 3 or self.steering_samples % 2 == 0:
            raise ValueError("steering_samples must be an odd integer >= 3")
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("map bounds must be ordered")
        if self.max_expansions < 1 or self.collision_radius < 0.0:
            raise ValueError("invalid expansion limit or collision radius")


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _distance(first: Pose, second: Pose) -> float:
    return float(np.hypot(first.x - second.x, first.y - second.y))


class HybridAStar:
    def __init__(
        self,
        config: HybridAStarConfig | None = None,
        obstacles: tuple[Rectangle, ...] = (),
    ):
        self.config = config or HybridAStarConfig()
        self.obstacles = tuple(obstacles)

    def _valid(self, pose: Pose) -> bool:
        c = self.config
        radius = c.collision_radius
        if not (
            c.x_min + radius <= pose.x <= c.x_max - radius
            and c.y_min + radius <= pose.y <= c.y_max - radius
        ):
            return False
        for obstacle in self.obstacles:
            if (
                obstacle.x - radius <= pose.x <= obstacle.x + obstacle.width + radius
                and obstacle.y - radius
                <= pose.y
                <= obstacle.y + obstacle.height + radius
            ):
                return False
        return True

    def _key(self, pose: Pose):
        c = self.config
        return (
            round((pose.x - c.x_min) / c.xy_resolution),
            round((pose.y - c.y_min) / c.xy_resolution),
            round(wrap_angle(pose.heading) / c.heading_resolution),
            int(pose.direction),
        )

    def _goal_reached(self, pose: Pose, goal: Pose) -> bool:
        return (
            _distance(pose, goal) <= self.config.goal_position_tolerance
            and abs(wrap_angle(pose.heading - goal.heading))
            <= self.config.goal_heading_tolerance
        )

    def _motion(self, pose: Pose, steering: float, direction: int):
        c = self.config
        steps = max(1, int(np.ceil(c.edge_length / (c.speed * c.integration_dt))))
        dt = c.edge_length / (c.speed * steps)
        velocity = direction * c.speed
        current = replace(pose, direction=direction)
        path = [current]
        for _ in range(steps):
            current = Pose(
                current.x + velocity * np.cos(current.heading) * dt,
                current.y + velocity * np.sin(current.heading) * dt,
                wrap_angle(
                    current.heading + velocity * np.tan(steering) / c.wheelbase * dt
                ),
                direction,
            )
            if not self._valid(current):
                return None
            path.append(current)
        return path

    def plan(self, start: Pose, goal: Pose) -> list[Pose] | None:
        """Return dense motion samples, or ``None`` when the search fails."""
        if not self._valid(start) or not self._valid(goal):
            return None
        if self._goal_reached(start, goal):
            return [replace(start, direction=1), replace(goal, direction=1)]

        steering_values = np.linspace(
            -self.config.max_steering,
            self.config.max_steering,
            self.config.steering_samples,
        )
        start = replace(start, heading=wrap_angle(start.heading), direction=1)
        start_key = self._key(start)
        counter = itertools.count()
        frontier = [
            (self._heuristic(start, goal), next(counter), 0.0, start_key, start)
        ]
        best_cost = {start_key: 0.0}
        parent = {start_key: None}
        edge_paths: dict[tuple, list[Pose]] = {}

        for _ in range(self.config.max_expansions):
            if not frontier:
                break
            _, _, cost, key, pose = heapq.heappop(frontier)
            if cost > best_cost.get(key, np.inf) + 1e-12:
                continue
            if self._goal_reached(pose, goal):
                return self._reconstruct(start, key, parent, edge_paths)

            for steering in steering_values:
                for direction in (1, -1):
                    segment = self._motion(pose, steering, direction)
                    if segment is None:
                        continue
                    successor = segment[-1]
                    successor_key = self._key(successor)
                    step_cost = self.config.edge_length
                    if direction < 0:
                        step_cost *= self.config.reverse_cost_factor
                    if direction != pose.direction:
                        step_cost += self.config.direction_switch_cost
                    new_cost = cost + step_cost
                    if new_cost >= best_cost.get(successor_key, np.inf):
                        continue
                    best_cost[successor_key] = new_cost
                    parent[successor_key] = key
                    edge_paths[successor_key] = segment
                    heapq.heappush(
                        frontier,
                        (
                            new_cost + 3 * self._heuristic(successor, goal),
                            next(counter),
                            new_cost,
                            successor_key,
                            successor,
                        ),
                    )
        return None

    @staticmethod
    def _heuristic(pose: Pose, goal: Pose) -> float:
        return _distance(pose, goal)

    @staticmethod
    def _reconstruct(start, final_key, parent, edge_paths) -> list[Pose]:
        segments = []
        cursor = final_key
        while parent[cursor] is not None:
            segments.append(edge_paths[cursor][1:])
            cursor = parent[cursor]
        result = [start]
        for segment in reversed(segments):
            result.extend(segment)
        if len(result) > 1:
            result[0] = replace(result[0], direction=result[1].direction)
        return result


def path_array(path: list[Pose]) -> np.ndarray:
    """Convert planner output to ``[x, y, yaw, direction]`` rows."""
    if path is None or len(path) < 2:
        raise ValueError("Hybrid A* path must contain at least two poses")
    return np.asarray(
        [[point.x, point.y, point.heading, point.direction] for point in path],
        dtype=float,
    )
