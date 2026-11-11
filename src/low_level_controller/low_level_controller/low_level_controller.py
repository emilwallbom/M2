"""Starter node for the car's low-level velocity controller."""

import rclpy
from rclpy.node import Node


class LowLevelController(Node):
    """Provide a starting point for normalized-input-to-velocity mapping."""

    def __init__(self):
        super().__init__("low_level_controller")
        # TODO: Add a control input subscriber and velocity output publisher.
        # TODO: Map control input in [0, 1] to velocity using car calibration.
        self.get_logger().info(
            "Low-level controller started; mapping is not implemented."
        )


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
