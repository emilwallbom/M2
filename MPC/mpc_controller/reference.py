"""Prepare a finite-horizon reference from a geometric parking path.

The Hybrid A* interface is an ``N x 4`` array whose rows are
``[x_m, y_m, yaw_rad, direction]``. Direction is ``+1`` for forward and
``-1`` for reverse. A three-column path is accepted as forward-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IPHI, IV, IX, IY, NX


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Reference states plus path progress for the next MPC cycle."""

    states: np.ndarray
    progress_index: int
    goal_reached: bool


def _prepare_path(path) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("path must have shape (M, 3) or (M, 4)")
    if len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("path must contain at least two finite poses")

    if points.shape[1] == 3:
        points = np.column_stack((points, np.ones(len(points))))
    elif np.any(~np.isin(points[:, 3], (-1.0, 1.0))):
        raise ValueError("path direction must be either -1 or +1")

    # Repeated poses can mark a direction change. Keep the later row so its
    # direction becomes the outgoing direction at that pose.
    cleaned = [points[0].copy()]
    for point in points[1:]:
        if np.linalg.norm(point[:2] - cleaned[-1][:2]) <= 1e-9:
            cleaned[-1] = point.copy()
        else:
            cleaned.append(point.copy())
    points = np.asarray(cleaned)
    if len(points) < 2:
        raise ValueError("path must contain at least two distinct positions")

    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    points[:, 2] = np.unwrap(points[:, 2])
    return points, arc_length


def _project_progress(
    position: np.ndarray,
    points: np.ndarray,
    arc_length: np.ndarray,
    progress_index: int,
    max_search_points: int,
) -> float:
    """Project onto a bounded forward path portion to avoid branch jumps."""
    first = int(np.clip(progress_index, 0, len(points) - 2))
    last = min(len(points) - 1, first + max_search_points)
    best_distance = np.inf
    best_progress = arc_length[first]

    for index in range(first, last):
        start = points[index, :2]
        segment = points[index + 1, :2] - start
        length_squared = float(segment @ segment)
        fraction = np.clip(
            float((position - start) @ segment) / length_squared, 0.0, 1.0
        )
        projection = start + fraction * segment
        distance = float((position - projection) @ (position - projection))
        if distance < best_distance:
            best_distance = distance
            best_progress = arc_length[index] + fraction * np.sqrt(length_squared)
    return best_progress


def _path_value(points: np.ndarray, arc_length: np.ndarray, progress: float):
    x = np.interp(progress, arc_length, points[:, 0])
    y = np.interp(progress, arc_length, points[:, 1])
    yaw = np.interp(progress, arc_length, points[:, 2])
    index = min(
        np.searchsorted(arc_length, progress, side="right") - 1, len(points) - 1
    )
    direction = points[max(index, 0), 3]
    return x, y, yaw, direction


def build_reference(
    state,
    path,
    horizon: int,
    dt: float,
    *,
    progress_index: int = 0,
    target_speed: float = 0.4,
    max_deceleration: float = 0.8,
    max_lateral_acceleration: float = 0.5,
    max_search_points: int = 40,
    goal_position_tolerance: float = 0.05,
    goal_speed_tolerance: float = 0.03,
) -> ReferenceTrajectory:
    """Resample a path into ``[X, Y, yaw, signed speed]`` states.

    Sampling advances along path arc length. The speed reference slows for
    curvature, direction changes, and the final goal. One horizon sample is
    placed at every crossed direction change to request a stop before reversing.
    """
    current = np.asarray(state, dtype=float)
    if current.shape != (NX,) or not np.all(np.isfinite(current)):
        raise ValueError(f"state must be a finite vector with shape ({NX},)")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    positive_values = (dt, target_speed, max_deceleration, max_lateral_acceleration)
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("dt, speeds, and acceleration limits must be positive")
    if max_search_points < 1:
        raise ValueError("max_search_points must be positive")

    points, arc_length = _prepare_path(path)
    points[:, 2] += 2.0 * np.pi * np.round(
        (current[IPHI] - points[0, 2]) / (2.0 * np.pi)
    )
    progress = _project_progress(
        current[[IX, IY]], points, arc_length, progress_index, max_search_points
    )

    direction_changes = np.flatnonzero(points[1:, 3] != points[:-1, 3]) + 1
    stop_positions = np.append(arc_length[direction_changes], arc_length[-1])
    yaw_gradient = np.gradient(points[:, 2], arc_length)
    nominal_step = target_speed * dt
    reference = np.empty((horizon + 1, NX), dtype=float)
    sample_progress = progress

    for step_index in range(horizon + 1):
        x, y, yaw, direction = _path_value(points, arc_length, sample_progress)
        future_stops = stop_positions[stop_positions >= sample_progress - 1e-9]
        distance_to_stop = np.min(future_stops - sample_progress)
        stopping_speed = np.sqrt(
            max(0.0, 2.0 * max_deceleration * distance_to_stop)
        )
        curvature = abs(np.interp(sample_progress, arc_length, yaw_gradient))
        curve_speed = np.sqrt(max_lateral_acceleration / max(curvature, 1e-9))
        speed = min(target_speed, stopping_speed, curve_speed)
        reference[step_index] = (x, y, yaw, direction * speed)

        if step_index == horizon or sample_progress >= arc_length[-1]:
            continue
        next_progress = min(sample_progress + nominal_step, arc_length[-1])
        crossed_stops = stop_positions[
            (stop_positions > sample_progress + 1e-9)
            & (stop_positions < next_progress - 1e-9)
        ]
        if crossed_stops.size:
            next_progress = crossed_stops[0]
        sample_progress = next_progress

    new_progress_index = int(
        np.clip(
            np.searchsorted(arc_length, progress, side="right") - 1,
            0,
            len(points) - 2,
        )
    )
    goal_distance = np.linalg.norm(current[[IX, IY]] - points[-1, :2])
    goal_reached = (
        goal_distance <= goal_position_tolerance
        and abs(current[IV]) <= goal_speed_tolerance
    )
    return ReferenceTrajectory(reference, new_progress_index, goal_reached)
