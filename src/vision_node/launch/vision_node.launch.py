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
    # data_recording.py(캘리브레이션 촬영)가 set_tool("Tool Weight_2FG") + set_tcp("2FG_TCP")를
    # 걸어놓고 get_current_posx()로 로봇 pose를 기록했기 때문에, T_gripper2camera.npy는
    # link_6(flange)이 아니라 그리퍼 끝 TCP("2FG_TCP") 기준으로 캘리브레이션된 값이다.
    # 2026-07-08 실기에서 measure_tcp_offset.py(get_current_posx() vs
    # get_current_tool_flange_posx() 비교)로 link_6->2FG_TCP를 직접 측정한 결과 x,y는
    # 노이즈 수준(0에 가까움), 회전 없이 z축으로만 228.6mm(그리퍼 길이) 떨어져 있음을 확인.
    # 이 오프셋 없이 T_gripper2camera.npy를 곧장 link_6에 물리면 z가 228.6mm만큼
    # 어긋난다(x,y는 카메라가 대략 아래를 보고 있어 영향이 작음).
    gripper_tcp_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.2286',
            '--frame-id', 'link_6', '--child-frame-id', 'gripper_tcp',
        ],
    )
    # hand-eye 캘리브레이션 결과(T_gripper2camera.npy, gripper_tcp -> 카메라 광학 좌표계)를
    # m 단위 평행이동 + 쿼터니언으로 변환해 반영 (src/vision_node/resource/T_gripper2camera.npy 참고).
    # 2026-07-08 실기 검증 중 발견: dsr_description2(m0609) URDF에 "flange" 프레임이
    # 없어(joint_6/link_6에서 체인이 끝남) robot_control의 base_link->link_6 방송과
    # 안 이어졌었다("TF has two or more unconnected trees") - link_6으로 정정.
    #
    # child-frame-id는 RealSense가 자체 발행하는 camera_link가 아니라 별도 이름
    # (camera_optical_calib)을 쓴다 - T_gripper2camera.npy는 이미 카메라 광학 좌표계
    # 기준으로 캘리브레이션된 값(verify.py 참고)인데, child-frame-id를 camera_link로
    # 하면 vision_node가 color_msg.header.frame_id(camera_color_optical_frame)로 TF를
    # 조회할 때 RealSense 내장 camera_link->camera_color_optical_frame 회전이 한 번 더
    # 걸려 캘리브레이션 회전이 중복 적용된다(x/z 축 결합 현상의 원인이었음).
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.032594', '--y', '0.065706', '--z', '-0.203569',
            '--qx', '0.000794', '--qy', '0.011346', '--qz', '0.999927', '--qw', '0.004194',
            '--frame-id', 'gripper_tcp', '--child-frame-id', 'camera_optical_calib',
        ],
    )
    vision_node = Node(
        package='vision_node',
        executable='vision_node',
        output='screen',
    )
    # object_detection(팀원3) 역할 - YOLO 추론만 하고 /detection/tool_boxes로 발행한다.
    # vision_node와 발행 토픽이 겹치지 않으므로(예전엔 둘 다 /vision/tool_track에 발행해
    # 동시 실행 금지였음) 함께 띄울 수 있다.
    tool_detection_node = Node(
        package='vision_node',
        executable='tool_detection_node',
    )
    return LaunchDescription(
        [realsense_launch, gripper_tcp_tf, static_tf, vision_node, tool_detection_node])
