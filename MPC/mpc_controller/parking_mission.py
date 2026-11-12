"""Publish one parking goal relative to the measured vehicle pose."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .parking_geometry import relative_goal


def yaw_from_odometry(message: Odometry) -> float:
    """Extract planar yaw from an odometry quaternion."""
    q = message.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y**2 + q.z**2),
    )


class ParkingMission(Node):
    """Wait for stationary odometry, then issue one relative parking target."""

    def __init__(self):
        super().__init__("parking_mission")
        self.declare_parameter("forward_m", -0.8)
        self.declare_parameter("left_m", 0.4)
        self.declare_parameter("yaw_delta_rad", 0.0)
        self.declare_parameter("start_delay_sec", 8.0)
        self.declare_parameter("stationary_speed_mps", 0.05)
        self.publisher = self.create_publisher(PoseStamped, "goal_pose", 1)
        self.subscription = self.create_subscription(Odometry, "odom", self._odom, 1)
        self.latest_odometry = None
        self.timer = None

    def _odom(self, message: Odometry) -> None:
        self.latest_odometry = message
        if self.timer is None:
            delay = float(self.get_parameter("start_delay_sec").value)
            self.timer = self.create_timer(delay, self._try_start)
            self.get_logger().info(
                f"Odometry received; parking mission armed with {delay:.1f}s delay"
            )

    def _try_start(self) -> None:
        speed_limit = float(self.get_parameter("stationary_speed_mps").value)
        if abs(self.latest_odometry.twist.twist.linear.x) > speed_limit:
            self.get_logger().warning("Waiting for the vehicle to become stationary")
            return

        forward = float(self.get_parameter("forward_m").value)
        left = float(self.get_parameter("left_m").value)
        yaw_delta = float(self.get_parameter("yaw_delta_rad").value)
        current_pose = self.latest_odometry.pose.pose
        x, y, yaw = relative_goal(
            current_pose.position.x,
            current_pose.position.y,
            yaw_from_odometry(self.latest_odometry),
            forward,
            left,
            yaw_delta,
        )
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.latest_odometry.header.frame_id
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self.publisher.publish(goal)
        self.get_logger().info(
            f"Parking mission started: body offset forward={forward:.2f} m, "
            f"left={left:.2f} m, yaw_delta={yaw_delta:.2f} rad"
        )
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ParkingMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
