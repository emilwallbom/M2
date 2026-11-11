"""Objective from Zhang et al., equation (20).

The Hybrid A* path deliberately does not appear in this module. It is an
initial guess for the nonlinear solver, not a trajectory to track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import NU


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for time, input effort, and input-rate effort."""

    time: float = 1.0
    steering: float = 0.01
    acceleration: float = 0.5
    steering_rate: float = 0.1
    acceleration_rate: float = 0.1

    def __post_init__(self):
        values = np.asarray(
            [self.time, self.steering, self.acceleration,
             self.steering_rate, self.acceleration_rate]
        )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("objective weights must be finite and nonnegative")


def trajectory_objective(
    controls,
    sample_time,
    previous_control=None,
    *,
    weights: ObjectiveWeights | None = None,
    xp=np,
):
    """Evaluate ``q*N*T + sum(u'Ru + du'Rbar*du)`` from equation (20)."""
    weights = weights or ObjectiveWeights()
    if xp is np:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[1] != NU or len(controls) < 1:
            raise ValueError(f"controls must have shape (N, {NU}) with N >= 1")
        if not np.isfinite(sample_time) or sample_time <= 0.0:
            raise ValueError("sample_time must be finite and positive")
    previous = controls[0, :] * 0.0 if previous_control is None else previous_control
    if xp is np:
        previous = np.asarray(previous, dtype=float)
        if previous.shape != (NU,) or not np.all(np.isfinite(previous)):
            raise ValueError(f"previous_control must have shape ({NU},)")

    previous_steering = previous[0]
    previous_acceleration = previous[1]
    objective = weights.time * controls.shape[0] * sample_time
    for index in range(controls.shape[0]):
        steering = controls[index, 0]
        acceleration = controls[index, 1]
        steering_rate = (steering - previous_steering) / sample_time
        acceleration_rate = (acceleration - previous_acceleration) / sample_time
        objective += (
            weights.steering * steering**2
            + weights.acceleration * acceleration**2
            + weights.steering_rate * steering_rate**2
            + weights.acceleration_rate * acceleration_rate**2
        )
        previous_steering = steering
        previous_acceleration = acceleration
    return objective


MPCWeights = ObjectiveWeights
