"""Ordinary state and input constraints for the trajectory optimizer.

OBCA environment constraints are separate; these limits apply with no obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IA, IDELTA, IV, NU, NX, VehicleParams


@dataclass(frozen=True)
class ConstraintSet:
    """Expressions and bounds in solver form: lower <= values <= upper."""

    values: object
    lower: np.ndarray
    upper: np.ndarray
    labels: tuple[str, ...]

    def is_satisfied(self, tolerance: float = 1e-9) -> bool:
        values = np.asarray(self.values, dtype=float).reshape(-1)
        return bool(
            np.all(values >= self.lower - tolerance)
            and np.all(values <= self.upper + tolerance)
        )


def _stack(xp, values):
    if hasattr(xp, "vertcat"):
        return xp.vertcat(*values)
    return np.asarray(values, dtype=float)


def build_constraints(
    states,
    controls,
    previous_control,
    dt,
    params: VehicleParams | None = None,
    *,
    xp=np,
) -> ConstraintSet:
    """Build speed, steering, acceleration, and steering-rate constraints."""
    params = params or VehicleParams()
    if xp is np:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        previous_control = np.asarray(previous_control, dtype=float)
        if states.ndim != 2 or states.shape[1] != NX:
            raise ValueError(f"states must have shape (N+1, {NX})")
        if controls.shape != (len(states) - 1, NU):
            raise ValueError(f"controls must have shape (N, {NU})")
        if previous_control.shape != (NU,):
            raise ValueError(f"previous_control must have shape ({NU},)")

    values = []
    lower = []
    upper = []
    labels = []

    for index in range(states.shape[0]):
        values.append(states[index, IV])
        lower.append(params.min_speed)
        upper.append(params.max_speed)
        labels.append(f"speed[{index}]")

    for index in range(controls.shape[0]):
        values.extend((controls[index, IDELTA], controls[index, IA]))
        lower.extend((-params.max_steering, -params.max_accel))
        upper.extend((params.max_steering, params.max_accel))
        labels.extend((f"steering[{index}]", f"acceleration[{index}]"))

        prior_steering = (
            previous_control[IDELTA]
            if index == 0
            else controls[index - 1, IDELTA]
        )
        values.append((controls[index, IDELTA] - prior_steering) / dt)
        lower.append(-params.max_steering_rate)
        upper.append(params.max_steering_rate)
        labels.append(f"steering_rate[{index}]")

    return ConstraintSet(
        _stack(xp, values),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        tuple(labels),
    )
