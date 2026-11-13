"""Side-by-side demo in turtlesim.

Left turtle (turtle1):  naive robot, acts on every decoder output (slow if p >= 0.5).
Right turtle (turtle2): confidence-aware robot with smoothing (slow if mean p >= 0.8, normal if <= 0.2).
Pen colour: green = normal speed, orange = cautious, red = slow.

    ros2 launch eeg_workload_robot compare.launch.py subject:=2 rate_scale:=1.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    subject = LaunchConfiguration("subject")
    rate_scale = LaunchConfiguration("rate_scale")
    loop = LaunchConfiguration("loop")
    return LaunchDescription([
        DeclareLaunchArgument("subject", default_value="2", description="person to replay (0-35)"),
        DeclareLaunchArgument("rate_scale", default_value="1.0", description="1.0 = real time, 4.0 = 4x faster"),
        DeclareLaunchArgument("loop", default_value="false"),
        Node(package="turtlesim", executable="turtlesim_node", name="sim"),
        Node(package="eeg_workload_robot", executable="speed_controller", name="naive_robot", output="screen",
             parameters=[{"policy": "naive", "turtle_name": "turtle1", "start_x": 2.5, "start_y": 4.0}]),
        Node(package="eeg_workload_robot", executable="speed_controller", name="aware_robot", output="screen",
             parameters=[{"policy": "smooth", "turtle_name": "turtle2", "start_x": 7.5, "start_y": 4.0}]),
        # start the decoder stream a few seconds later, so both turtles are ready
        TimerAction(period=4.0, actions=[
            Node(package="eeg_workload_robot", executable="decoder_replay", name="decoder_replay", output="screen",
                 parameters=[{"subject": subject, "rate_scale": rate_scale, "loop": loop}]),
        ]),
    ])
