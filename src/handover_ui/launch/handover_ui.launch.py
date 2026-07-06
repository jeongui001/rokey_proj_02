from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ui_node = Node(package='handover_ui', executable='handover_ui', output='screen')
    return LaunchDescription([ui_node])
