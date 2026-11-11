"""ROS 2 interface scaffold; intentionally publishes no actuator commands."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from cascar_interfaces.msg import ActuatorCommand

from mpc_controller.controller import MPCController


class MPCNode(Node):
    """Receive ROS inputs and reserve the command interface for future MPC."""

    def __init__(self):
        super().__init__("mpc_controller")
        self.declare_parameter("control_rate_hz", 20.0)
        control_rate = float(self.get_parameter("control_rate_hz").value)
        if not 0.0 < control_rate < float("inf"):
            raise ValueError("control_rate_hz must be finite and positive")

        self.controller = MPCController()
        self.latest_odometry = None
        self.latest_reference_path = None
        self.odometry_subscription = self.create_subscription(
            Odometry, "odom", self._on_odometry, 10
        )
        self.path_subscription = self.create_subscription(
            Path, "reference_path", self._on_reference_path, 10
        )
        self.command_publisher = self.create_publisher(
            ActuatorCommand, "actuator_command", 1
        )
        self.control_timer = self.create_timer(1.0 / control_rate, self._control_tick)
        self.get_logger().info(
            "MPC scaffold started. Optimization is not implemented; "
            "no actuator commands will be published."
        )

    def _on_odometry(self, message: Odometry):
        self.latest_odometry = message

    def _on_reference_path(self, message: Path):
        self.latest_reference_path = message

    def _control_tick(self):
        if self.latest_odometry is None or self.latest_reference_path is None:
            return
        # TODO: validate timestamps, frames, finite values and nonempty paths.
        # TODO: convert ROS messages into model state and numeric path data.
        # TODO: build the horizon reference and call self.controller.solve().
        # TODO: handle failed/stale solutions and define stopping behavior.
        # TODO: map physical inputs to normalized throttle/steering in [-1, 1],
        # stamp an ActuatorCommand and publish only after validation.
        # The scaffold must not produce placeholder driving commands.


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPCNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
