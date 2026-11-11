"""Launch the MPC interface in the same namespace as the vehicle backend."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("mpc_controller")) / "config" / "mpc.yaml"
    )
    return LaunchDescription([
        DeclareLaunchArgument("agent_name", default_value="panther"),
        DeclareLaunchArgument("params_file", default_value=default_config),
        Node(
            package="mpc_controller",
            executable="mpc_node",
            name="mpc_controller",
            namespace=LaunchConfiguration("agent_name"),
            parameters=[LaunchConfiguration("params_file")],
            output="screen",
        ),
    ])
