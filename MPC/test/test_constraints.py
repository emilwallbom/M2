import numpy as np

from mpc_controller.constraints import build_constraints
from mpc_controller.model import VehicleParams


def test_constraints_accept_values_inside_limits():
    states = np.zeros((4, 4))
    controls = np.array([[0.02, 0.1], [0.04, 0.0], [0.02, -0.1]])
    constraints = build_constraints(
        states, controls, np.zeros(2), 0.1, VehicleParams()
    )
    assert constraints.is_satisfied()


def test_constraints_detect_steering_rate_violation():
    states = np.zeros((2, 4))
    controls = np.array([[0.2, 0.0]])
    constraints = build_constraints(
        states, controls, np.zeros(2), 0.1, VehicleParams()
    )
    assert not constraints.is_satisfied()
    assert "steering_rate[0]" in constraints.labels


def test_constraints_detect_speed_violation():
    states = np.zeros((2, 4))
    states[1, 3] = 1.1
    constraints = build_constraints(
        states, np.zeros((1, 2)), np.zeros(2), 0.1, VehicleParams()
    )
    assert not constraints.is_satisfied()
