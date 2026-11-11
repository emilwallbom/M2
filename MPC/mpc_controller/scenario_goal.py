"""Publish one deterministic goal for the full simulated control chain."""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class ScenarioGoal(Node):
    """Send a goal shortly after simulator odometry becomes available."""

    def __init__(self):
        super().__init__("scenario_goal")
        self.declare_parameter("goal_x", 1.5)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("delay_sec", 3.0)
        self.publisher = self.create_publisher(PoseStamped, "goal_pose", 1)
        self.subscription = self.create_subscription(
            Odometry, "odom", self._on_odometry, 1
        )
        self.frame_id = ""
        self.timer = None

    def _on_odometry(self, message: Odometry) -> None:
        if self.timer is not None:
            return
        self.frame_id = message.header.frame_id
        delay = float(self.get_parameter("delay_sec").value)
        self.timer = self.create_timer(delay, self._publish_goal)
        self.get_logger().info(f"Odometry received; publishing test goal in {delay:.1f}s")

    def _publish_goal(self) -> None:
        if self.publisher.get_subscription_count() == 0:
            self.get_logger().info("Waiting for the planner goal subscription")
            return
        yaw = float(self.get_parameter("goal_yaw").value)
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.pose.position.x = float(self.get_parameter("goal_x").value)
        message.pose.position.y = float(self.get_parameter("goal_y").value)
        message.pose.orientation.z = math.sin(yaw / 2.0)
        message.pose.orientation.w = math.cos(yaw / 2.0)
        self.publisher.publish(message)
        self.get_logger().info(
            f"Published test goal ({message.pose.position.x:.2f}, "
            f"{message.pose.position.y:.2f}, {yaw:.2f}) in {self.frame_id}"
        )
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioGoal()
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
