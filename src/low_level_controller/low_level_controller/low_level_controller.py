"""Starter node for the car's low-level velocity controller."""

import math

import rclpy
from cascar_interfaces.msg import ActuatorCommand
from nav_msgs.msg import Odometry
from rclpy.node import Node


class LowLevelController(Node):
    """Provide a starting point for normalized-input-to-velocity mapping."""

    def __init__(self):
        super().__init__("low_level_controller")
        self.odom: Odometry | None = None
        self.velocity: float | None = None
        self.odom_subscription = self.create_subscription(
            Odometry,
            "odom",
            self.odom_callback,
            1,
        )
        self.control_publisher = self.create_publisher(
            ActuatorCommand,
            "actuator_command",
            1,
        )

        self.dt: float = 0.02
        self.control_timer = self.create_timer(self.dt, self.control_callback)
        self.reference_velocity: float = 0.5
        self.reference_steering: float = 0.0
        self.k: float = 1.0
        self.previous_error: float = 0.0
        self.previous_u: float = 0.0
        self.ti: float = 1.0
        self.max_steering_angle: float = math.radians(30.0)

    def control_callback(self) -> None:
        """Control the vehicle based on the current state."""
        if self.velocity is None:
            return

        error: float = self.reference_velocity - self.velocity
        delta_u = self.k * (error - self.previous_error + self.dt / self.ti * error)

        if self.previous_u + delta_u > 1.0:
            u = 1.0
        elif self.previous_u + delta_u < -1.0:
            u = -1.0
        else:
            u = self.previous_u + delta_u

        self.previous_u = u
        self.previous_error = error

        u_steering = self.reference_steering / self.max_steering_angle
        self.publish_control(u, u_steering)

    def odom_callback(self, msg: Odometry) -> None:
        """Store odometry and signed forward velocity in m/s."""
        self.odom = msg
        self.velocity = msg.twist.twist.linear.x

    def publish_control(self, throttle: float, steering: float = 0.0) -> None:
        """Publish normalized actuator efforts in [-1, 1], not velocity targets."""
        msg = ActuatorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = float(throttle)
        msg.steering = float(steering)
        self.control_publisher.publish(msg)


def main(args=None):
    """Run the low-level controller node."""
    rclpy.init(args=args)
    node = LowLevelController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
