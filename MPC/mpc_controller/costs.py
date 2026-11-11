"""Tracking and control costs for the nonlinear MPC problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import IA, IDELTA, IPHI, IV, IX, IY, NU, NX
from .reference import ReferenceTrajectory


@dataclass(frozen=True)
class MPCWeights:
    """Scalar quadratic weights in SI units."""

    position: float = 10.0
    heading: float = 5.0
    speed: float = 2.0
    steering: float = 0.1
    acceleration: float = 0.1
    steering_rate: float = 1.0
    acceleration_rate: float = 0.1
    terminal_factor: float = 5.0


def angle_error(angle, reference, xp=np):
    """Shortest signed angular difference across the +/-pi boundary."""
    difference = angle - reference
    return xp.arctan2(xp.sin(difference), xp.cos(difference))


def tracking_cost(
    states,
    controls,
    reference,
    previous_control=None,
    *,
    dt: float = 1.0,
    weights: MPCWeights | None = None,
    xp=np,
):
    """Return the scalar finite-horizon tracking objective.

    Trajectories use row-major shapes ``states=(N+1, 4)`` and
    ``controls=(N, 2)`` with NumPy and CasADi.
    """
    weights = weights or MPCWeights()
    reference_states = (
        reference.states if isinstance(reference, ReferenceTrajectory) else reference
    )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    if xp is np:
        states = np.asarray(states, dtype=float)
        controls = np.asarray(controls, dtype=float)
        reference_states = np.asarray(reference_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != NX:
            raise ValueError(f"states must have shape (N+1, {NX})")
        if controls.shape != (len(states) - 1, NU):
            raise ValueError(f"controls must have shape (N, {NU})")
        if reference_states.shape != states.shape:
            raise ValueError("reference states must have the same shape as states")

    previous = controls[0, :] * 0.0 if previous_control is None else previous_control
    objective = 0.0
    final_index = states.shape[0] - 1
    for index in range(states.shape[0]):
        dx = states[index, IX] - reference_states[index, IX]
        dy = states[index, IY] - reference_states[index, IY]
        dphi = angle_error(
            states[index, IPHI], reference_states[index, IPHI], xp
        )
        dv = states[index, IV] - reference_states[index, IV]
        factor = weights.terminal_factor if index == final_index else 1.0
        objective += factor * (
            weights.position * (dx * dx + dy * dy)
            + weights.heading * dphi * dphi
            + weights.speed * dv * dv
        )

    for index in range(controls.shape[0]):
        steering = controls[index, IDELTA]
        acceleration = controls[index, IA]
        delta_steering = (steering - previous[IDELTA]) / dt
        delta_acceleration = (acceleration - previous[IA]) / dt
        objective += (
            weights.steering * steering * steering
            + weights.acceleration * acceleration * acceleration
            + weights.steering_rate * delta_steering * delta_steering
            + weights.acceleration_rate * delta_acceleration * delta_acceleration
        )
        previous = controls[index, :]
    return objective
