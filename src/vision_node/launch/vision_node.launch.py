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
            # realsense2_camera 기본값(camera_namespace='camera')은 노드 이름과 겹쳐
            # 토픽이 /camera/camera/color/image_raw로 발행된다 - vision_node/
            # tool_detection_node는 /camera/color/image_raw(단일)를 구독하므로 비워서 맞춘다
            # (2026-07-08 실기 검증 중 발견: 이걸 안 하면 두 노드 다 이미지를 하나도 못 받는다).
            'camera_namespace': '',
        }.items()
    )
    # hand-eye 캘리브레이션 결과(T_gripper2camera.npy, link_6 -> camera_link)를
    # m 단위 평행이동 + 쿼터니언으로 변환해 반영 (src/vision_node/resource/T_gripper2camera.npy 참고).
    # 2026-07-08 실기 검증 중 발견: dsr_description2(m0609) URDF에 "flange" 프레임이
    # 없어(joint_6/link_6에서 체인이 끝남) robot_control의 base_link->link_6 방송과
    # 안 이어졌었다("TF has two or more unconnected trees") - link_6으로 정정.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.032594', '--y', '0.065706', '--z', '-0.203569',
            '--qx', '0.000794', '--qy', '0.011346', '--qz', '0.999927', '--qw', '0.004194',
            '--frame-id', 'link_6', '--child-frame-id', 'camera_link',
        ],
    )
    vision_node = Node(
        package='vision_node',
        executable='vision_node',
    )
    # object_detection(팀원3) 역할 - YOLO 추론만 하고 /detection/tool_boxes로 발행한다.
    # vision_node와 발행 토픽이 겹치지 않으므로(예전엔 둘 다 /vision/tool_track에 발행해
    # 동시 실행 금지였음) 함께 띄울 수 있다.
    tool_detection_node = Node(
        package='vision_node',
        executable='tool_detection_node',
    )
    return LaunchDescription([realsense_launch, static_tf, vision_node, tool_detection_node])
