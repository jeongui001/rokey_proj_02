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
            # align_depth.enable(+enable_sync)를 켜서 드라이버가 정렬해주게 했었으나,
            # 이 카메라/드라이버 조합(FW 5.13.0.50, ROS wrapper v4.57.7)에서 이 조합 자체가
            # 근본적으로 깨져 있음을 2026-07-12 실기에서 확인했다 - 켜면 depth 프레임이
            # 요청 fps의 정확히 2배로 나오고 aligned_depth_to_color가 항상 0프레임만
            # 발행한다(해상도/fps를 바꿔도 재현, rs-hello-realsense로 depth 센서 자체는
            # 정상임을 별도 확인). GitHub에도 realsense-ros/librealsense에서 enable_sync+
            # align_depth 조합이 여러 버전에 걸쳐 반복 보고된 알려진 문제. 그래서 드라이버
            # 정렬은 끄고, vision_node가 raw depth(/camera/depth/image_rect_raw)를 받아
            # grasp_geometry.align_depth_to_color로 직접 컬러 픽셀 격자에 정렬한다
            # (vision_node.py의 _align_depth_msg/_get_depth_to_color_extrinsics 참고).
            'align_depth.enable': 'false',
            'enable_sync': 'false',
            # realsense2_camera 기본값(camera_namespace='camera')은 노드 이름과 겹쳐
            # 토픽이 /camera/camera/color/image_raw로 발행된다 - vision_node/
            # tool_detection_node는 /camera/color/image_raw(단일)를 구독하므로 비워서 맞춘다
            # (2026-07-08 실기 검증 중 발견: 이걸 안 하면 두 노드 다 이미지를 하나도 못 받는다).
            'camera_namespace': '',
            # depth post-processing 필터(2026-07-10, "z가 계속 내려간다" 조사 중 추가) -
            # patch_median_depth(공간)/ToolTracker의 EMA(시간)는 우리 코드 안에서 노이즈를
            # 누르지만, 드라이버가 애초에 내놓는 depth 원본을 다듬는 게 아니라 하류에서만
            # 대응하는 것. RealSense 공식 필터는 더 상류(센서 출력 직후)에서 적용돼 이후
            # 모든 단계에 도움이 된다.
            'spatial_filter.enable': 'true',   # 프레임 내 edge-preserving 스무딩(홀/스펙클 감소)
            'temporal_filter.enable': 'true',  # 프레임간 노이즈 저감(우리가 겪는 노이즈 층위와 정확히 일치)
            # decimation_filter는 추가 안 함 - 이미 424x240으로 낮춘 해상도를 더 줄이면
            # keypoint 정밀도 손실 우려. hole_filling_filter도 보류 - patch_median_depth+
            # depth_valid 플래그로 무효 구간을 이미 명시적으로 다루고 있어 중복이고,
            # 잘못 채워진 값이 유효로 오인될 위험이 있음.
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
        # 마지막 안전망 - _safe_call이 대부분의 프레임 단위 예외를 이제 막아주지만,
        # 그래도 예측 못한 예외로 프로세스가 죽으면 respawn 없이는 카메라 송출이
        # 영구히 멈춘다(2026-07-12 확인). 몇 초 뒤 자동 재기동되게 한다.
        respawn=True,
        respawn_delay=2.0,
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
