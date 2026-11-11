"""Vehicle dynamics, independent of ROS.

Kinematic bicycle model (Zhang, Liniger, Borrelli, "Optimization-Based
Collision Avoidance", Sec. V-B, eq. (19)), referenced to the rear axle.

State  x = [X, Y, phi, v]   (m, m, rad, m/s), rear-axle centre, world frame
Input  u = [delta, a]       (rad, m/s^2), steering angle and acceleration

Cascar's actuator command is a dimensionless effort in [-1, 1]; converting a
planned (delta, a, v) to that effort is the tracker's job, not this module's.

All functions take an array backend ``xp``. The default is numpy for
simulation and checks. Pass ``casadi`` to build the symbolic NLP, e.g.
``step(x, u, T, xp=casadi)``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# State / input indices.
IX, IY, IPHI, IV = 0, 1, 2, 3
IDELTA, IA = 0, 1
NX, NU = 4, 2

_VEHICLE_YAML = (
    Path(__file__).resolve().parents[2]
    / "src/provided/cascar_interfaces/config/vehicle_parameters.yaml"
)


@dataclass(frozen=True)
class VehicleParams:
    """Vehicle constants and limits (SI units)."""

    wheelbase: float = 0.275  # L
    max_steering: float = 0.35  # |delta| limit
    max_steering_rate: float = 0.6  # |d delta/dt| limit, rad/s. TODO: measure
    max_accel: float = 1.0
    min_speed: float = -1.0  # reversing is needed for parking
    max_speed: float = 1.0

    # Footprint rectangle B = {y : G y <= g}, in the body frame (origin at the
    # rear axle, x forward). PLACEHOLDERS estimated from cascar.xacro (wheels
    # at x=0 and x=L, track ~0.27 m); measure the real car and update.
    length: float = 0.40
    width: float = 0.27
    rear_overhang: float = 0.06  # rear axle to rear bumper

    @property
    def center_offset(self):
        """Distance from the rear axle to the footprint centre, along heading."""
        return self.length / 2.0 - self.rear_overhang


def load_vehicle_params(path=_VEHICLE_YAML, **overrides) -> VehicleParams:
    """Build VehicleParams from vehicle_parameters.yaml (+ keyword overrides)."""
    import yaml

    with open(path) as f:
        p = yaml.safe_load(f)["/**"]["ros__parameters"]
    values = dict(
        wheelbase=p["wheelbase_m"],
        max_steering=p["max_steering_angle_rad"],
        max_accel=p["max_acceleration_mps2"],
        max_speed=p["max_speed_mps"],
        min_speed=-p["max_speed_mps"],
    )
    values.update(overrides)
    return VehicleParams(**values)


def _stack(xp, items):
    """Stack scalars into a column vector for numpy or casadi."""
    if hasattr(xp, "vertcat"):
        return xp.vertcat(*items)
    return np.array(items, dtype=float)


def continuous_dynamics(x, u, params: VehicleParams, xp=np):
    """x_dot = f~(x, u), eq. (19)."""
    phi, v = x[IPHI], x[IV]
    delta, a = u[IDELTA], u[IA]
    return _stack(
        xp,
        [
            v * xp.cos(phi),
            v * xp.sin(phi),
            v * xp.tan(delta) / params.wheelbase,
            a,
        ],
    )


def step(x, u, T, params: VehicleParams, xp=np):
    """Forward-Euler step x_{k+1} = x_k + T f~(x_k, u_k).

    ``T`` may be a number or a decision variable (the paper optimizes T_opt).
    """
    return x + T * continuous_dynamics(x, u, params, xp)


# Kept for the scaffold's original interface.
def predict_next_state(state, control, dt, params: VehicleParams = None):
    """Predict one time step using the vehicle model."""
    return step(state, control, dt, params or VehicleParams())


def rollout(x0, U, T, params: VehicleParams = None):
    """Simulate forward from x0 with inputs U (N x NU); returns (N+1) x NX."""
    params = params or VehicleParams()
    X = [np.asarray(x0, dtype=float)]
    for u in U:
        X.append(step(X[-1], np.asarray(u, dtype=float), T, params))
    return np.array(X)


def rotation(x, xp=np):
    """R(x): body-to-world rotation (2x2) from the yaw angle."""
    c, s = xp.cos(x[IPHI]), xp.sin(x[IPHI])
    if hasattr(xp, "vertcat"):
        return xp.vertcat(xp.horzcat(c, -s), xp.horzcat(s, c))
    return np.array([[c, -s], [s, c]])


def translation(x, params: VehicleParams, xp=np):
    """t(x): world position of the footprint centre (rear axle + offset)."""
    c = params.center_offset
    return _stack(
        xp,
        [x[IX] + c * xp.cos(x[IPHI]), x[IY] + c * xp.sin(x[IPHI])],
    )


def footprint_halfspaces(params: VehicleParams):
    """(G, g) with B = {y : G y <= g}, centred at the origin (body frame).

    Together with R(x), t(x) this gives E(x) = R(x) B + t(x), eq. (5b).
    """
    G = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    g = np.array(
        [params.length / 2, params.length / 2, params.width / 2, params.width / 2]
    )
    return G, g


def input_bounds(params: VehicleParams):
    """Box bounds (lower, upper) on u = [delta, a] and on v."""
    lo = np.array([-params.max_steering, -params.max_accel])
    hi = np.array([params.max_steering, params.max_accel])
    return lo, hi, (params.min_speed, params.max_speed)
