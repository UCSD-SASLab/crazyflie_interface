from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("backend", default_value="sim"),
            DeclareLaunchArgument("uri", default_value=""),
            DeclareLaunchArgument("run_input", default_value="True"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("crazyflie_interface"), "launch", "launch.py"]
                        )
                    ]
                ),
                launch_arguments={
                    "backend": LaunchConfiguration("backend"),
                    "uri": LaunchConfiguration("uri"),
                }.items(),
            ),
            Node(
                package="crazyflie_interface",
                executable="realmcf_tracker_controller.py",
                name="realmcf_tracker_controller",
                output="screen",
            ),
            Node(
                package="crazyflie_interface",
                executable="input_lemniscate.py",
                name="input_lemniscate",
                output="screen",
                condition=IfCondition(LaunchConfiguration("run_input")),
            ),
        ]
    )
