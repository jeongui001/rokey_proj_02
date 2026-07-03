from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource([
            FindPackageShare('rosbridge_server'), '/launch/rosbridge_websocket_launch.xml'
        ])
    )
    ui_process = ExecuteProcess(cmd=['handover_ui'], output='screen')
    return LaunchDescription([rosbridge_launch, ui_process])
