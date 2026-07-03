from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'depth_module.profile': '424x240x60',
            'rgb_camera.profile': '424x240x60',
        }.items()
    )
    # NOTE: realsense2_camera 버전에 따라 launch 인자명이 다를 수 있으니
    # 설치된 realsense-ros 문서로 재확인할 것.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # TODO: hand-eye 캘리브레이션 결과값(flange -> camera_link)으로 인자 교체
        arguments=['0', '0', '0', '0', '0', '0', 'flange', 'camera_link'],
    )
    vision_node = Node(
        package='vision_node',
        executable='vision_node',
    )
    return LaunchDescription([realsense_launch, static_tf, vision_node])
