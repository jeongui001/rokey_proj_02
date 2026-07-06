import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    task_manager_params = os.path.join(
        get_package_share_directory('task_manager'), 'config', 'task_manager_params.yaml')
    task_manager_node = Node(
        package='task_manager', executable='task_manager_node',
        parameters=[task_manager_params])

    stt_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('stt_node'), '/launch/stt_node.launch.py'
        ])
    )
    vision_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('vision_node'), '/launch/vision_node.launch.py'
        ])
    )
    robot_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('robot_control'), '/launch/robot_control.launch.py'
        ])
    )
    operator_gui_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('operator_gui'), '/launch/operator_gui.launch.py'
        ])
    )

    return LaunchDescription([
        task_manager_node,
        stt_node_launch,
        vision_node_launch,
        robot_control_launch,
        operator_gui_launch,
    ])
