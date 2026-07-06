from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rosbridge_host_arg = DeclareLaunchArgument(
        'rosbridge_host', default_value='localhost',
        description='operator_gui가 접속할 rosbridge WebSocket host.')
    rosbridge_port_arg = DeclareLaunchArgument(
        'rosbridge_port', default_value='9090',
        description='operator_gui가 접속할 rosbridge WebSocket port.')
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic', default_value='/vision/debug_image/compressed',
        description='카메라 압축 영상(sensor_msgs/CompressedImage) 토픽 이름.')
    camera_stale_timeout_s_arg = DeclareLaunchArgument(
        'camera_stale_timeout_s', default_value='2.0',
        description='이 시간(초) 동안 새 영상이 없으면 "카메라 영상이 멈췄습니다"로 표시.')

    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            FindPackageShare('rosbridge_server'), '/launch/rosbridge_websocket_launch.xml'
        ])
    )
    ui_process = ExecuteProcess(
        cmd=['operator_gui'],
        additional_env={
            'OPERATOR_GUI_ROSBRIDGE_HOST': LaunchConfiguration('rosbridge_host'),
            'OPERATOR_GUI_ROSBRIDGE_PORT': LaunchConfiguration('rosbridge_port'),
            'OPERATOR_GUI_CAMERA_TOPIC': LaunchConfiguration('camera_topic'),
            'OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S': LaunchConfiguration('camera_stale_timeout_s'),
        },
        output='screen')
    return LaunchDescription([
        rosbridge_host_arg,
        rosbridge_port_arg,
        camera_topic_arg,
        camera_stale_timeout_s_arg,
        rosbridge_launch,
        ui_process,
    ])
