"""Starter node for the car's low-level velocity controller."""

import csv
import math
from datetime import datetime
from pathlib import Path

import rclpy
from cascar_interfaces.msg import ActuatorCommand, OptimizedTrajectory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .trajectory_sampling import sample_reference


class LowLevelController(Node):
    """Provide a starting point for normalized-input-to-velocity mapping."""

    def __init__(self):
        super().__init__("low_level_controller")
        self.odom: Odometry | None = None
        self.velocity: float | None = None
        self.declare_parameter("speed_log_directory", "speed_logs")
        self.declare_parameter("max_throttle_output", 1.0)
        self.speed_log_path = (
            Path(self.get_parameter("speed_log_directory").value).expanduser()
            / f"odom_speed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        ).resolve()
        self.speed_samples: list[tuple] = []
        self.control_samples: list[tuple] = []
        self.last_control: tuple = (None, None, None, None)
        self.control_log_path = self.speed_log_path.with_name(
            self.speed_log_path.stem + "_controls.csv"
        )
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
        trajectory_qos = QoSProfile(depth=1)
        trajectory_qos.reliability = ReliabilityPolicy.RELIABLE
        trajectory_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.trajectory_subscription = self.create_subscription(
            OptimizedTrajectory,
            "optimized_trajectory",
            self.trajectory_callback,
            trajectory_qos,
        )

        self.dt: float = 0.02
        self.control_timer = self.create_timer(self.dt, self.control_callback)
        self.trajectory: OptimizedTrajectory | None = None
        self.trajectory_start_ns: int = 0
        self.reference_velocity: float = 0.0
        self.reference_steering: float = 0.0
        self.k: float = 1.0
        self.previous_error: float = 0.0
        self.previous_u: float = 0.0
        self.ti: float = 1.0
        self.max_left_steering_angle: float = math.radians(25.0)
        self.max_right_steering_angle: float = math.radians(35.0)
        self.steering_offset: float = math.radians(5.0)
        self.max_throttle_output = max(
            0.0,
            min(1.0, float(self.get_parameter("max_throttle_output").value)),
        )

    def control_callback(self) -> None:
        """Control the vehicle based on the current state."""
        if self.velocity is None or self.trajectory is None:
            return

        elapsed = (
            self.get_clock().now().nanoseconds - self.trajectory_start_ns
        ) * 1e-9
        self.reference_velocity, self.reference_steering, _ = sample_reference(
            self.trajectory.velocities,
            self.trajectory.steering_angles,
            self.trajectory.sample_time,
            elapsed,
        )

        error: float = self.reference_velocity - self.velocity
        delta_u = self.k * (error - self.previous_error + self.dt / self.ti * error)

        if self.previous_u + delta_u > self.max_throttle_output:
            u = self.max_throttle_output
        elif self.previous_u + delta_u < -self.max_throttle_output:
            u = -self.max_throttle_output
        else:
            u = self.previous_u + delta_u

        self.previous_u = u
        self.previous_error = error

        if self.reference_steering >= 0:
            # Left steering
            u_steering = (
                self.reference_steering - self.steering_offset
            ) / self.max_left_steering_angle
        else:
            # Right steering
            u_steering = (
                self.reference_steering + self.steering_offset
            ) / self.max_right_steering_angle

        self.publish_control(u, max(-1.0, min(1.0, u_steering)))

    def trajectory_callback(self, msg: OptimizedTrajectory) -> None:
        """Accept a complete trajectory produced by the OBCA planner."""
        controls = len(msg.poses) - 1
        values = [
            msg.sample_time,
            *msg.velocities,
            *msg.steering_angles,
            *msg.accelerations,
        ]
        if (
            len(msg.poses) < 2
            or len(msg.velocities) != len(msg.poses)
            or len(msg.steering_angles) != controls
            or len(msg.accelerations) != controls
            or not all(math.isfinite(value) for value in values)
            or msg.sample_time <= 0.0
        ):
            self.get_logger().error("Rejected malformed optimized trajectory")
            return
        self.trajectory = msg
        self.trajectory_start_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(
            f"Following optimized trajectory with {len(msg.poses)} states"
        )

    def odom_callback(self, msg: Odometry) -> None:
        """Store odometry and signed forward velocity in m/s."""
        self.odom = msg
        self.velocity = msg.twist.twist.linear.x
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        self.speed_samples.append((
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            linear.x,
            linear.y,
            linear.z,
            math.sqrt(linear.x**2 + linear.y**2 + linear.z**2),
            angular.x,
            angular.y,
            angular.z,
            *self.last_control,
        ))

    def save_speed_log(self) -> None:
        """Save odometry and every published control command to CSV files."""
        self.speed_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.speed_log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "stamp_sec",
                "stamp_nanosec",
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
                "speed_m_s",
                "wx_rad_s",
                "wy_rad_s",
                "wz_rad_s",
                "control_stamp_sec",
                "control_stamp_nanosec",
                "throttle",
                "steering",
            ])
            writer.writerows(self.speed_samples)
        with self.control_log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "stamp_sec",
                "stamp_nanosec",
                "throttle",
                "steering",
                "reference_velocity_m_s",
                "reference_steering_rad",
            ])
            writer.writerows(self.control_samples)
        print(
            f"Saved {len(self.speed_samples)} odometry speed samples "
            f"to {self.speed_log_path}",
            flush=True,
        )
        print(
            f"Saved {len(self.control_samples)} control samples "
            f"to {self.control_log_path}",
            flush=True,
        )

    def publish_control(self, throttle: float, steering: float = 0.0) -> None:
        """Publish normalized actuator efforts in [-1, 1], not velocity targets."""
        msg = ActuatorCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = float(throttle)
        msg.steering = float(steering)
        self.control_publisher.publish(msg)
        self.last_control = (
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            msg.throttle,
            msg.steering,
        )
        self.control_samples.append((
            *self.last_control,
            self.reference_velocity,
            self.reference_steering,
        ))


def main(args=None):
    """Run the low-level controller node."""
    rclpy.init(args=args)
    node = LowLevelController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.save_speed_log()
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
