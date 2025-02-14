from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='armor_check',
            executable='display',
            name='display',
            output='screen'
        ),
        Node(
            package='armor_check',
            executable='yolo',
            name='yolo',
            output='screen'
        ),
        Node(
            package='armor_check',
            executable='traditional',
            name='traditional',
            output='screen'
        )
    ])