"""Set a robot's speed from the EEG workload decoder.

Subscribes:  /workload/probability (Float32), /workload/label (Int32), /workload/session_done (Bool)
Publishes:   <cmd_vel_topic> geometry_msgs/Twist (10 Hz), /<name>/speed_scale (Float32), /<name>/mode (String)

The robot drives in a circle whose speed follows the policy (normal 1.0, cautious 0.7, slow 0.4).
In turtlesim the pen colour shows the mode: green = normal, orange = cautious, red = slow.

Parameters:
  policy          naive | aware | smooth (default smooth)
  turtle_name     turtlesim turtle to drive (default turtle1); empty = not using turtlesim
  cmd_vel_topic   default /<turtle_name>/cmd_vel (set e.g. /cmd_vel for a TurtleBot3 in Gazebo)
  start_x, start_y  turtlesim start position (turtle1 is teleported, other names are spawned)
  base_linear, base_angular  speeds at scale 1.0 (radius = linear / angular stays constant)
  log_dir         where a CSV of every decision is written (default ~/eeg_robot_logs)
"""
import csv
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String

from eeg_workload_robot.policy import SessionStats, SpeedPolicy

PEN = {"normal": (40, 170, 90), "cautious": (240, 160, 40), "slow": (220, 50, 50)}


class SpeedController(Node):
    def __init__(self):
        super().__init__("speed_controller")
        self.declare_parameter("policy", "smooth")
        self.declare_parameter("turtle_name", "turtle1")
        self.declare_parameter("cmd_vel_topic", "")
        self.declare_parameter("start_x", 5.5)
        self.declare_parameter("start_y", 3.0)
        self.declare_parameter("base_linear", 1.5)
        self.declare_parameter("base_angular", 1.0)
        self.declare_parameter("log_dir", os.path.expanduser("~/eeg_robot_logs"))

        self.policy_name = self.get_parameter("policy").value
        self.policy = SpeedPolicy(self.policy_name)
        self.stats = SessionStats()
        name = str(self.get_parameter("turtle_name").value or "")
        self.turtle = "" if name.lower() in ("", "none") else name   # "none" = not using turtlesim
        topic = self.get_parameter("cmd_vel_topic").value or f"/{self.turtle or 'robot'}/cmd_vel"
        self.ns = self.turtle or "robot"
        self.mode, self.scale, self.label, self.p = "normal", 1.0, -1, float("nan")

        self.pub_cmd = self.create_publisher(Twist, topic, 10)
        self.pub_scale = self.create_publisher(Float32, f"/{self.ns}/speed_scale", 10)
        self.pub_mode = self.create_publisher(String, f"/{self.ns}/mode", 10)
        self.create_subscription(Int32, "/workload/label", self.on_label, 10)
        self.create_subscription(Float32, "/workload/probability", self.on_prob, 10)
        self.create_subscription(Bool, "/workload/session_done", self.on_done, 10)
        self.create_timer(0.1, self.drive)

        log_dir = self.get_parameter("log_dir").value
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{self.ns}_{self.policy_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.log_f = open(self.log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["window", "true_label", "p_high_workload", "mode", "speed_scale"])
        self.pen_client = None
        self.get_logger().info(f"[{self.ns}] policy={self.policy_name}, publishing {topic}")

    # ---------------------------------------------------------------- turtlesim setup
    def setup_turtlesim(self):
        if not self.turtle:
            return
        from turtlesim.srv import SetPen, Spawn, TeleportAbsolute
        x, y = float(self.get_parameter("start_x").value), float(self.get_parameter("start_y").value)

        def call(srv_type, name, req):
            cli = self.create_client(srv_type, name)
            if not cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(f"service {name} not available; is turtlesim running?")
                return None
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            return fut.result()

        if self.turtle == "turtle1":
            call(SetPen, "/turtle1/set_pen", SetPen.Request(off=1))
            call(TeleportAbsolute, "/turtle1/teleport_absolute", TeleportAbsolute.Request(x=x, y=y, theta=0.0))
        else:
            call(Spawn, "/spawn", Spawn.Request(x=x, y=y, theta=0.0, name=self.turtle))
        self.pen_client = self.create_client(SetPen, f"/{self.turtle}/set_pen")
        self.set_pen("normal")

    def set_pen(self, mode):
        if self.pen_client is None or not self.pen_client.service_is_ready():
            return
        from turtlesim.srv import SetPen
        r, g, b = PEN[mode]
        self.pen_client.call_async(SetPen.Request(r=r, g=g, b=b, width=4, off=0))

    # ---------------------------------------------------------------- callbacks
    def on_label(self, msg):
        self.label = int(msg.data)

    def on_prob(self, msg):
        self.p = float(msg.data)
        mode, scale = self.policy.step(self.p)
        if mode != self.mode:
            self.set_pen(mode)
        self.mode, self.scale = mode, scale
        self.stats.update(mode, self.label)
        self.pub_scale.publish(Float32(data=scale))
        self.pub_mode.publish(String(data=mode))
        self.log.writerow([self.stats.n - 1, self.label, round(self.p, 4), mode, scale])
        self.log_f.flush()
        truth = "OVERLOADED" if self.label == 1 else "rest"
        flag = "  <-- full speed while overloaded!" if (self.label == 1 and mode == "normal") else ""
        self.get_logger().info(f"[{self.ns}/{self.policy_name}] person={truth:10s} p={self.p:.2f} -> {mode:8s}{flag}")

    def on_done(self, msg):
        if msg.data:
            self.get_logger().info(f"[{self.ns}/{self.policy_name}] SESSION SUMMARY {self.stats.summary()}  log: {self.log_path}")
            self.policy.reset()
            self.stats = SessionStats()

    def drive(self):
        lin = float(self.get_parameter("base_linear").value) * self.scale
        ang = float(self.get_parameter("base_angular").value) * self.scale
        t = Twist()
        t.linear.x, t.angular.z = lin, ang
        self.pub_cmd.publish(t)

    def destroy_node(self):
        try:
            self.pub_cmd.publish(Twist())   # stop the robot
            self.log_f.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeedController()
    try:
        node.setup_turtlesim()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
