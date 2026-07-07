from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic', default_value='/vision/debug_image/compressed',
        description='카메라 압축 영상(sensor_msgs/CompressedImage) 토픽 이름.')
    camera_stale_timeout_s_arg = DeclareLaunchArgument(
        'camera_stale_timeout_s', default_value='2.0',
        description='이 시간(초) 동안 새 영상이 없으면 "카메라 영상이 멈췄습니다"로 표시.')

    ui_process = ExecuteProcess(
        cmd=['ros2', 'run', 'operator_gui', 'operator_gui'],
        additional_env={
            'OPERATOR_GUI_CAMERA_TOPIC': LaunchConfiguration('camera_topic'),
            'OPERATOR_GUI_CAMERA_STALE_TIMEOUT_S': LaunchConfiguration('camera_stale_timeout_s'),
        },
        output='screen')
    return LaunchDescription([
        camera_topic_arg,
        camera_stale_timeout_s_arg,
        ui_process,
    ])
