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
    default_local_params = os.path.join(
        get_package_share_directory('robot_control'), 'config',
        'robot_control_local_params.yaml')

    hardware_enabled_arg = DeclareLaunchArgument(
        'hardware_enabled', default_value='true',
        description='true로 설정해야만 실제 M0609/RG2에 명령을 전송한다 (기본값: dry_run).')
    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='named_poses 등 robot_control 파라미터 YAML 경로.')
    calibration_params_file_arg = DeclareLaunchArgument(
        'calibration_params_file', default_value=default_calibration_params,
        description=(
            'params_file 위에 override할 실측 캘리브레이션 YAML 경로 '
            '(handover_hold 당김 감지값 등). params_file 뒤에 로드되어 같은 키를 덮어쓴다.'))
    local_params_file_arg = DeclareLaunchArgument(
        'local_params_file', default_value=default_local_params,
        description=(
            '이 컴퓨터에서만 다른, 팀과 공유하면 안 되는 값(예: doosan-robot2 포크 차이로 '
            '인한 doosan_driver.controller_name) override용 개인 YAML 경로. 기본값은 '
            '빈 파일(robot_control_local_params.yaml)이라 아무것도 바꾸지 않는다 - '
            '개인 값은 커밋되지 않는 dev/ 아래 파일을 만들어 이 인자로 넘긴다.'))

    robot_control_node = Node(
        package='robot_control',
        executable='robot_control_node',
        parameters=[
            LaunchConfiguration('params_file'),
            LaunchConfiguration('calibration_params_file'),
            LaunchConfiguration('local_params_file'),
            {'hardware_enabled': LaunchConfiguration('hardware_enabled')},
        ],
    )

    return LaunchDescription([
        hardware_enabled_arg,
        params_file_arg,
        calibration_params_file_arg,
        local_params_file_arg,
        robot_control_node,
    ])
