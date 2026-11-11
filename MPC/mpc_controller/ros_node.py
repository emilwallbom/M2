"""ROS 2 pipeline: odometry + goal -> Hybrid A* -> OBCA trajectory."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from cascar_interfaces.msg import OptimizedTrajectory
from geometry_msgs.msg import Pose as PoseMessage
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .controller import OptimizerConfig, TrajectoryOptimizer
from .costs import ObjectiveWeights
from .hybrid_astar import (
    HybridAStar,
    HybridAStarConfig,
    Pose,
    Rectangle,
    path_array,
)
from .model import VehicleParams
from .obstacles import axis_aligned_rectangle
from .reference import build_initial_guess


def _yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2),
    )


def _quaternion_from_yaw(yaw: float):
    quaternion = PoseMessage().orientation
    quaternion.z = math.sin(yaw / 2.0)
    quaternion.w = math.cos(yaw / 2.0)
    return quaternion


def _parse_rectangles(values) -> tuple[Rectangle, ...]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return ()
    if values.ndim != 1 or values.size % 4:
        raise ValueError("obstacles must be flat [x, y, width, height, ...]")
    return tuple(Rectangle(*row) for row in values.reshape(-1, 4))


class MPCNode(Node):
    """Run one complete planning solve whenever a new goal is received."""

    def __init__(self):
        super().__init__("mpc_controller")
        self._declare_parameters()
        self.params = VehicleParams()
        rectangles = _parse_rectangles(self.get_parameter("obstacles").value)
        map_bounds = list(self.get_parameter("map_bounds").value)
        if len(map_bounds) != 4:
            raise ValueError("map_bounds must be [x_min, x_max, y_min, y_max]")

        minimum_clearance = float(self.get_parameter("minimum_clearance_m").value)
        collision_radius = math.hypot(
            max(self.params.length - self.params.rear_overhang, self.params.rear_overhang),
            self.params.width / 2.0,
        ) + minimum_clearance
        astar_config = HybridAStarConfig(
            wheelbase=self.params.wheelbase,
            max_steering=self.params.max_steering,
            steering_samples=int(self.get_parameter("astar_steering_samples").value),
            speed=float(self.get_parameter("astar_speed_mps").value),
            edge_length=float(self.get_parameter("astar_edge_length_m").value),
            integration_dt=float(self.get_parameter("astar_integration_dt_sec").value),
            xy_resolution=float(self.get_parameter("astar_xy_resolution_m").value),
            heading_resolution=float(self.get_parameter("astar_heading_resolution_rad").value),
            goal_position_tolerance=float(self.get_parameter("astar_goal_position_tolerance_m").value),
            goal_heading_tolerance=float(self.get_parameter("astar_goal_heading_tolerance_rad").value),
            max_expansions=int(self.get_parameter("astar_max_expansions").value),
            x_min=float(map_bounds[0]),
            x_max=float(map_bounds[1]),
            y_min=float(map_bounds[2]),
            y_max=float(map_bounds[3]),
            collision_radius=collision_radius,
        )
        self.astar = HybridAStar(astar_config, rectangles)

        optimizer_config = OptimizerConfig(
            minimum_sample_time=float(self.get_parameter("minimum_sample_time_sec").value),
            maximum_sample_time=float(self.get_parameter("maximum_sample_time_sec").value),
            minimum_clearance=minimum_clearance,
            map_bounds=tuple(float(v) for v in map_bounds),
            solver_max_iterations=int(self.get_parameter("solver_max_iterations").value),
            solver_max_cpu_time=float(self.get_parameter("solver_max_cpu_time_sec").value),
        )
        weights = ObjectiveWeights(
            time=float(self.get_parameter("objective_time_weight").value),
            steering=float(self.get_parameter("objective_steering_weight").value),
            acceleration=float(self.get_parameter("objective_acceleration_weight").value),
            steering_rate=float(self.get_parameter("objective_steering_rate_weight").value),
            acceleration_rate=float(self.get_parameter("objective_acceleration_rate_weight").value),
        )
        obca_obstacles = tuple(
            axis_aligned_rectangle(
                rectangle.x + rectangle.width / 2.0,
                rectangle.y + rectangle.height / 2.0,
                rectangle.width,
                rectangle.height,
                label=f"obstacle_{index}",
            )
            for index, rectangle in enumerate(rectangles)
        )
        self.optimizer = TrajectoryOptimizer(
            optimizer_config, self.params, weights, obca_obstacles
        )
        self.initial_sample_time = float(
            self.get_parameter("initial_sample_time_sec").value
        )
        self.warm_start_spacing = float(
            self.get_parameter("warm_start_spacing_m").value
        )
        self.goal_speed = float(self.get_parameter("goal_speed_mps").value)

        output_qos = QoSProfile(depth=1)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        output_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.warm_start_publisher = self.create_publisher(
            Path, "warm_start_path", output_qos
        )
        self.optimized_path_publisher = self.create_publisher(
            Path, "optimized_path", output_qos
        )
        self.trajectory_publisher = self.create_publisher(
            OptimizedTrajectory, "optimized_trajectory", output_qos
        )
        self.odometry_subscription = self.create_subscription(
            Odometry, "odom", self._on_odometry, 10
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped, "goal_pose", self._on_goal, 10
        )
        planning_rate = float(self.get_parameter("planning_rate_hz").value)
        if planning_rate <= 0.0:
            raise ValueError("planning_rate_hz must be positive")
        self.timer = self.create_timer(1.0 / planning_rate, self._plan_if_ready)
        self.latest_odometry = None
        self.pending_goal = None
        self.planning = False
        self.get_logger().info(
            "Planner ready: publish PoseStamped goal_pose after odometry is available"
        )

    def _declare_parameters(self):
        defaults = {
            "planning_rate_hz": 2.0,
            "initial_sample_time_sec": 0.15,
            "warm_start_spacing_m": 0.04,
            "goal_speed_mps": 0.0,
            "minimum_sample_time_sec": 0.05,
            "maximum_sample_time_sec": 0.35,
            "minimum_clearance_m": 0.05,
            "objective_time_weight": 1.0,
            "objective_steering_weight": 0.01,
            "objective_acceleration_weight": 0.5,
            "objective_steering_rate_weight": 0.1,
            "objective_acceleration_rate_weight": 0.1,
            "solver_max_iterations": 500,
            "solver_max_cpu_time_sec": 10.0,
            "astar_steering_samples": 9,
            "astar_speed_mps": 0.35,
            "astar_edge_length_m": 0.25,
            "astar_integration_dt_sec": 0.05,
            "astar_xy_resolution_m": 0.08,
            "astar_heading_resolution_rad": float(np.deg2rad(10.0)),
            "astar_goal_position_tolerance_m": 0.12,
            "astar_goal_heading_tolerance_rad": float(np.deg2rad(10.0)),
            "astar_max_expansions": 50_000,
            "map_bounds": [-5.0, 5.0, -5.0, 5.0],
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self.declare_parameter("obstacles", Parameter.Type.DOUBLE_ARRAY)

    def _on_odometry(self, message: Odometry):
        self.latest_odometry = message

    def _on_goal(self, message: PoseStamped):
        self.pending_goal = message
        self.get_logger().info("New goal received; scheduling Hybrid A* and OBCA")

    def _plan_if_ready(self):
        if self.planning or self.latest_odometry is None or self.pending_goal is None:
            return
        self.planning = True
        goal_message = self.pending_goal
        self.pending_goal = None
        try:
            self._plan(self.latest_odometry, goal_message)
        except (RuntimeError, ValueError) as error:
            self.get_logger().error(f"Planning failed: {error}")
        finally:
            self.planning = False

    def _plan(self, odometry: Odometry, goal_message: PoseStamped):
        odom_frame = odometry.header.frame_id
        goal_frame = goal_message.header.frame_id
        if odom_frame and goal_frame and odom_frame != goal_frame:
            raise ValueError(
                f"frame mismatch: odometry={odom_frame}, goal={goal_frame}"
            )
        frame_id = goal_frame or odom_frame or "map"
        start_pose = odometry.pose.pose
        goal_pose = goal_message.pose
        start_speed = float(np.clip(
            odometry.twist.twist.linear.x,
            self.params.min_speed,
            self.params.max_speed,
        ))
        start = Pose(
            start_pose.position.x,
            start_pose.position.y,
            _yaw_from_quaternion(start_pose.orientation),
        )
        goal = Pose(
            goal_pose.position.x,
            goal_pose.position.y,
            _yaw_from_quaternion(goal_pose.orientation),
        )
        astar_path = self.astar.plan(start, goal)
        if astar_path is None:
            raise RuntimeError("Hybrid A* found no path within its expansion limit")

        guess = build_initial_guess(
            path_array(astar_path),
            sample_time=self.initial_sample_time,
            params=self.params,
            start_speed=start_speed,
            goal_speed=self.goal_speed,
            minimum_point_spacing=self.warm_start_spacing,
        )
        start_state = np.array([start.x, start.y, start.heading, start_speed])
        goal_state = np.array([goal.x, goal.y, goal.heading, self.goal_speed])
        result = self.optimizer.solve(start_state, goal_state, guess)
        if not result.success:
            raise RuntimeError(f"OBCA returned {result.status}")

        stamp = self.get_clock().now().to_msg()
        self.warm_start_publisher.publish(
            self._path_message(guess.states, frame_id, stamp)
        )
        self.optimized_path_publisher.publish(
            self._path_message(result.states, frame_id, stamp)
        )
        trajectory = OptimizedTrajectory()
        trajectory.header.frame_id = frame_id
        trajectory.header.stamp = stamp
        trajectory.sample_time = result.sample_time
        trajectory.poses = [self._pose_message(state) for state in result.states]
        trajectory.velocities = result.states[:, 3].tolist()
        trajectory.steering_angles = result.controls[:, 0].tolist()
        trajectory.accelerations = result.controls[:, 1].tolist()
        trajectory.objective = result.objective
        self.trajectory_publisher.publish(trajectory)
        self.get_logger().info(
            f"Published {len(result.states)} states: A*={len(astar_path)} samples, "
            f"Topt={result.sample_time:.3f}s, duration={result.total_time:.2f}s, "
            f"solve={result.solve_time:.2f}s"
        )

    @staticmethod
    def _pose_message(state) -> PoseMessage:
        pose = PoseMessage()
        pose.position.x = float(state[0])
        pose.position.y = float(state[1])
        pose.orientation = _quaternion_from_yaw(float(state[2]))
        return pose

    @classmethod
    def _path_message(cls, states, frame_id, stamp) -> Path:
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = stamp
        for state in states:
            stamped = PoseStamped()
            stamped.header = path.header
            stamped.pose = cls._pose_message(state)
            path.poses.append(stamped)
        return path


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
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
