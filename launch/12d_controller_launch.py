from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import yaml
import os
from ament_index_python.packages import get_package_share_directory

def parse_yaml(context):
    crazyflies_yaml = LaunchConfiguration('crazyflies_yaml').perform(context)
    with open(crazyflies_yaml, 'r') as ymlfile:
        crazyflies = yaml.safe_load(ymlfile)
    parameters = [crazyflies]
    return [
        Node(
            package='crazyflie_interface',
            executable='pursuit_evasion_controller.py',
            name='pursuit_evasion_controller',
            output='screen',
            parameters=parameters
        ),
    ]

def generate_launch_description():
    default_crazyflies_yaml_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'crazyflies.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'crazyflies_yaml', default_value=default_crazyflies_yaml_path,
            description='Path to crazyflies.yaml'
        ),
        DeclareLaunchArgument(
            'backend', default_value='sim',
            description='cpp / cflib / sim backend for crazyflie (crazyswarm2)'
        ),
        DeclareLaunchArgument(
            'uri', default_value='',
            description='Robot number'
        ),
        OpaqueFunction(function=parse_yaml),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('crazyflie_interface'),
                    'launch',
                    'launch.py'
                ]),
            ]),
            launch_arguments={
                'backend': LaunchConfiguration('backend'),
                'uri': LaunchConfiguration('uri')
            }.items()
        ) 
    ]) 

















