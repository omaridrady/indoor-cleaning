"""Run the RPLIDAR C1 driver plus the motion recorder.

    ros2 launch ~/lidar_motion/lidar_motion.launch.py                 # driver + recorder
    ros2 launch ~/lidar_motion/lidar_motion.launch.py driver:=false   # driver already running
    ros2 launch ~/lidar_motion/lidar_motion.launch.py out:=~/run1
Needs `source ~/ros2_ws/install/setup.bash` for the sllidar_ros2 driver.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))

C1_PORT = ('/dev/serial/by-id/usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_'
           '827e2745221ef111a50cc4e40f0f12f8-if00-port0')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('driver', default_value='true',
                              description='start the lidar driver (false if already running)'),
        DeclareLaunchArgument('serial_port', default_value=C1_PORT),
        DeclareLaunchArgument('out', default_value='',
                              description='bag directory (default ~/lidar_bags/motion_<time>)'),

        Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            output='screen',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': LaunchConfiguration('serial_port'),
                'serial_baudrate': 460800,
                'frame_id': 'laser',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
            condition=IfCondition(LaunchConfiguration('driver')),
        ),

        ExecuteProcess(
            # One "--out=<value>" token, so an empty value can't leave --out dangling.
            cmd=['python3', os.path.join(HERE, 'lidar_motion_recorder.py'),
                 ['--out=', LaunchConfiguration('out')]],
            output='screen',
        ),
    ])
