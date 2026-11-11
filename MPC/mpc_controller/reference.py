"""Convert a Hybrid A* path into an NLP initial guess.

This module does no reference tracking. Its output is used only with
``Opti.set_initial`` and therefore does not constrain or penalize the final
optimized trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IA, IDELTA, IPHI, IV, NU, NX, VehicleParams


@dataclass(frozen=True)
class TrajectoryGuess:
    """Initial state/input values supplied to the nonlinear optimizer."""

    states: np.ndarray
    controls: np.ndarray
    sample_time: float

    def __post_init__(self):
        states = np.asarray(self.states, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        if states.ndim != 2 or states.shape[1] != NX or len(states) < 2:
            raise ValueError(f"states must have shape (N+1, {NX})")
        if controls.shape != (len(states) - 1, NU):
            raise ValueError(f"controls must have shape (N, {NU})")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
            raise ValueError("initial guess must contain only finite values")
        if not np.isfinite(self.sample_time) or self.sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)

    @property
    def horizon(self) -> int:
        return len(self.controls)


def _validate_path(path) -> np.ndarray:
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("path must have rows [x, y, yaw] or [x, y, yaw, direction]")
    if len(points) < 2 or not np.all(np.isfinite(points)):
        raise ValueError("path must contain at least two finite poses")
    if points.shape[1] == 3:
        points = np.column_stack((points, np.ones(len(points))))
    if np.any(~np.isin(points[:, 3], (-1.0, 1.0))):
        raise ValueError("path direction must be +1 or -1")
    if np.all(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) <= 1e-9):
        raise ValueError("path must contain at least two distinct positions")
    points = points.copy()
    points[:, IPHI] = np.unwrap(points[:, IPHI])
    return points


def build_initial_guess(
    path,
    *,
    sample_time: float = 0.15,
    params: VehicleParams | None = None,
    start_speed: float = 0.0,
    goal_speed: float = 0.0,
    minimum_point_spacing: float = 0.04,
) -> TrajectoryGuess:
    """Infer velocity and input guesses from Hybrid A* poses."""
    params = params or VehicleParams()
    points = _validate_path(path)
    if not np.isfinite(sample_time) or sample_time <= 0.0:
        raise ValueError("sample_time must be finite and positive")
    if not np.isfinite(minimum_point_spacing) or minimum_point_spacing < 0.0:
        raise ValueError("minimum_point_spacing must be finite and nonnegative")

    # Hybrid A* returns dense integration samples. Keep direction switches and
    # thin nearby samples so the NLP horizon remains computationally practical.
    if minimum_point_spacing > 0.0 and len(points) > 2:
        retained = [0]
        for index in range(1, len(points) - 1):
            changes_direction = (
                points[index, 3] != points[index - 1, 3]
                or points[index, 3] != points[index + 1, 3]
            )
            far_enough = (
                np.linalg.norm(points[index, :2] - points[retained[-1], :2])
                >= minimum_point_spacing
            )
            if changes_direction or far_enough:
                retained.append(index)
        retained.append(len(points) - 1)
        points = points[np.unique(retained)]

    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    segment_speeds = points[:-1, 3] * segment_lengths / sample_time
    segment_speeds = np.clip(
        segment_speeds, params.min_speed, params.max_speed
    )
    states = np.zeros((len(points), NX))
    states[:, :3] = points[:, :3]
    states[:-1, IV] = segment_speeds
    states[0, IV] = start_speed
    states[-1, IV] = goal_speed

    controls = np.zeros((len(points) - 1, NU))
    yaw_change = np.diff(states[:, IPHI])
    denominator = np.where(
        np.abs(segment_speeds) > 1e-6,
        segment_speeds * sample_time,
        np.copysign(1e-6, segment_speeds + 1e-12),
    )
    controls[:, IDELTA] = np.arctan(params.wheelbase * yaw_change / denominator)
    controls[:, IDELTA] = np.clip(
        controls[:, IDELTA], -params.max_steering, params.max_steering
    )
    controls[:, IA] = np.diff(states[:, IV]) / sample_time
    controls[:, IA] = np.clip(controls[:, IA], -params.max_accel, params.max_accel)
    return TrajectoryGuess(states, controls, float(sample_time))


build_reference = build_initial_guess
