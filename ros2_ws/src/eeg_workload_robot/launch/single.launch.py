"""One robot with a chosen policy. Works with turtlesim (default) or any robot that listens to a Twist topic.

    ros2 launch eeg_workload_robot single.launch.py policy:=smooth subject:=2
    # for a TurtleBot3 already running in Gazebo (no turtlesim window):
    ros2 launch eeg_workload_robot single.launch.py use_turtlesim:=false cmd_vel_topic:=/cmd_vel
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    use_ts = LaunchConfiguration("use_turtlesim")
    return LaunchDescription([
        DeclareLaunchArgument("policy", default_value="smooth", description="naive | aware | smooth"),
        DeclareLaunchArgument("subject", default_value="2"),
        DeclareLaunchArgument("rate_scale", default_value="1.0"),
        DeclareLaunchArgument("use_turtlesim", default_value="true"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/turtle1/cmd_vel"),
        DeclareLaunchArgument("base_linear", default_value="1.5"),
        DeclareLaunchArgument("base_angular", default_value="1.0"),
        Node(package="turtlesim", executable="turtlesim_node", name="sim", condition=IfCondition(use_ts)),
        Node(package="eeg_workload_robot", executable="speed_controller", name="robot", output="screen",
             parameters=[{
                 "policy": LaunchConfiguration("policy"),
                 "turtle_name": PythonExpression(["'turtle1' if '", use_ts, "'.lower() == 'true' else 'none'"]),
                 "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                 "base_linear": LaunchConfiguration("base_linear"),
                 "base_angular": LaunchConfiguration("base_angular"),
             }]),
        TimerAction(period=4.0, actions=[
            Node(package="eeg_workload_robot", executable="decoder_replay", name="decoder_replay", output="screen",
                 parameters=[{"subject": LaunchConfiguration("subject"),
                              "rate_scale": LaunchConfiguration("rate_scale")}]),
        ]),
    ])
