"""Run the simulator, trajectory optimizer, and low-level controller together."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    agent_name = LaunchConfiguration("agent_name")
    visualization = LaunchConfiguration("visualization")
    mpc_config = PathJoinSubstitution(
        [FindPackageShare("mpc_controller"), "config", "mpc.yaml"]
    )
    cascar_launch = PathJoinSubstitution(
        [FindPackageShare("cascar"), "launch", "project_launch.py"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("agent_name", default_value="panther"),
        DeclareLaunchArgument("visualization", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cascar_launch),
            launch_arguments={
                "backend": "simulator",
                "agent_name": agent_name,
                "visualization": visualization,
                "qualisys_used": "false",
            }.items(),
        ),
        Node(
            package="mpc_controller",
            executable="mpc_node",
            name="mpc_controller",
            namespace=agent_name,
            parameters=[mpc_config],
            output="screen",
        ),
        Node(
            package="low_level_controller",
            executable="low_level_controller",
            name="low_level_controller",
            namespace=agent_name,
            output="screen",
        ),
        Node(
            package="mpc_controller",
            executable="scenario_goal",
            name="scenario_goal",
            namespace=agent_name,
            output="screen",
        ),
    ])
