"""ROS-independent geometry for a parking target relative to the car."""

import math


def relative_goal(x, y, yaw, forward, left, yaw_delta):
    """Transform a body-frame offset into an odometry-frame target pose."""
    goal_x = x + forward * math.cos(yaw) - left * math.sin(yaw)
    goal_y = y + forward * math.sin(yaw) + left * math.cos(yaw)
    return goal_x, goal_y, yaw + yaw_delta
