import math

import pytest
from mpc_controller.parking_geometry import relative_goal


def test_relative_goal_is_rotated_into_odometry_frame():
    yaw = math.pi / 2.0
    x, y, goal_yaw = relative_goal(
        1.0, 2.0, yaw, forward=-0.8, left=0.4, yaw_delta=0.1
    )

    assert x == pytest.approx(0.6)
    assert y == pytest.approx(1.2)
    assert goal_yaw == pytest.approx(yaw + 0.1)
