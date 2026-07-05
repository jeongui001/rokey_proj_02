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
            'depth_module.depth_profile': '424x240x60',
            'rgb_camera.color_profile': '424x240x60',
            'align_depth.enable': 'true',
        }.items()
    )
    # hand-eye 캘리브레이션 결과(T_gripper2camera.npy, flange -> camera_link)를
    # m 단위 평행이동 + 쿼터니언으로 변환해 반영 (src/vision_node/resource/T_gripper2camera.npy 참고)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.032594', '--y', '0.065706', '--z', '-0.203569',
            '--qx', '0.000794', '--qy', '0.011346', '--qz', '0.999927', '--qw', '0.004194',
            '--frame-id', 'flange', '--child-frame-id', 'camera_link',
        ],
    )
    vision_node = Node(
        package='vision_node',
        executable='vision_node',
    )
    return LaunchDescription([realsense_launch, static_tf, vision_node])
