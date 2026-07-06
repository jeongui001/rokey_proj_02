import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('task_manager'), 'config', 'task_manager_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='trigger/tools/auto.config_ready 등 task_manager 파라미터 YAML 경로.')

    return LaunchDescription([
        params_file_arg,
        Node(
            package='task_manager',
            executable='task_manager_node',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
