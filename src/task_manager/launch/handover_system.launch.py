import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    task_manager_params = os.path.join(
        get_package_share_directory('task_manager'), 'config', 'task_manager_params.yaml')
    task_manager_node = Node(
        package='task_manager', executable='task_manager_node',
        parameters=[task_manager_params])

    default_local_params = os.path.join(
        get_package_share_directory('robot_control'), 'config',
        'robot_control_local_params.yaml')
    local_params_file_arg = DeclareLaunchArgument(
        'local_params_file', default_value=default_local_params,
        description=(
            'robot_control.launch.py로 그대로 전달되는, 이 컴퓨터에서만 다른 개인 '
            'override YAML 경로(예: doosan-robot2 포크 차이로 인한 '
            'doosan_driver.controller_name). robot_control.launch.py의 '
            'local_params_file 인자 설명 참고.'))

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
        ]),
        launch_arguments={
            'local_params_file': LaunchConfiguration('local_params_file'),
        }.items(),
    )
    operator_gui_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('operator_gui'), '/launch/operator_gui.launch.py'
        ])
    )

    return LaunchDescription([
        local_params_file_arg,
        task_manager_node,
        stt_node_launch,
        vision_node_launch,
        robot_control_launch,
        operator_gui_launch,
    ])
