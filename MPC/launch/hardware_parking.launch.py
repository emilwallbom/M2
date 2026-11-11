"""Start the complete laptop-side stack and one physical parking mission."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    agent_name = LaunchConfiguration("agent_name")
    cascar_launch = PathJoinSubstitution(
        [FindPackageShare("cascar"), "launch", "project_launch.py"]
    )
    mpc_config = PathJoinSubstitution(
        [FindPackageShare("mpc_controller"), "config", "mpc.yaml"]
    )

    return LaunchDescription([
        DeclareLaunchArgument("agent_name", default_value="tiger"),
        DeclareLaunchArgument("visualization", default_value="true"),
        DeclareLaunchArgument("qualisys_used", default_value="true"),
        DeclareLaunchArgument("forward_m", default_value="-0.8"),
        DeclareLaunchArgument("left_m", default_value="0.4"),
        DeclareLaunchArgument("yaw_delta_rad", default_value="0.0"),
        DeclareLaunchArgument("start_delay_sec", default_value="8.0"),
        DeclareLaunchArgument("max_throttle_output", default_value="0.2"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(cascar_launch),
            launch_arguments={
                "backend": "vehicle",
                "agent_name": agent_name,
                "visualization": LaunchConfiguration("visualization"),
                "qualisys_used": LaunchConfiguration("qualisys_used"),
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
            parameters=[{
                "max_throttle_output": ParameterValue(
                    LaunchConfiguration("max_throttle_output"), value_type=float
                ),
            }],
            output="screen",
        ),
        Node(
            package="mpc_controller",
            executable="parking_mission",
            name="parking_mission",
            namespace=agent_name,
            parameters=[{
                "forward_m": ParameterValue(
                    LaunchConfiguration("forward_m"), value_type=float
                ),
                "left_m": ParameterValue(
                    LaunchConfiguration("left_m"), value_type=float
                ),
                "yaw_delta_rad": ParameterValue(
                    LaunchConfiguration("yaw_delta_rad"), value_type=float
                ),
                "start_delay_sec": ParameterValue(
                    LaunchConfiguration("start_delay_sec"), value_type=float
                ),
            }],
            output="screen",
        ),
    ])
