import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('robot_control'), 'config', 'robot_control_params.yaml')
    default_calibration_params = os.path.join(
        get_package_share_directory('robot_control'), 'config',
        'robot_control_calibration_params.yaml')

    hardware_enabled_arg = DeclareLaunchArgument(
        'hardware_enabled', default_value='false',
        description='true로 설정해야만 실제 M0609/RG2에 명령을 전송한다 (기본값: dry_run).')
    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='named_poses 등 robot_control 파라미터 YAML 경로.')
    calibration_params_file_arg = DeclareLaunchArgument(
        'calibration_params_file', default_value=default_calibration_params,
        description=(
            'params_file 위에 override할 실측 캘리브레이션 YAML 경로 '
            '(handover_hold 당김 감지값 등). params_file 뒤에 로드되어 같은 키를 덮어쓴다.'))

    robot_control_node = Node(
        package='robot_control',
        executable='robot_control_node',
        parameters=[
            LaunchConfiguration('params_file'),
            LaunchConfiguration('calibration_params_file'),
            {'hardware_enabled': LaunchConfiguration('hardware_enabled')},
        ],
    )

    return LaunchDescription([
        hardware_enabled_arg,
        params_file_arg,
        calibration_params_file_arg,
        robot_control_node,
    ])
