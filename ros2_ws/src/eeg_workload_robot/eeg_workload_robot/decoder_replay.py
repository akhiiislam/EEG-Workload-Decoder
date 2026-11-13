"""Replay the EEG workload decoder's outputs as a live ROS 2 stream.

Publishes, every window (2 s by default):
  /workload/probability  std_msgs/Float32  P(high workload) from the calibrated LOSO decoder
  /workload/label        std_msgs/Int32    true condition (0 rest, 1 arithmetic), for evaluation only
  /workload/session_done std_msgs/Bool     True after the last window of the session

Parameters:
  csv_path   (str)   default: <package share>/data/decoder_stream.csv
  subject    (int)   person to replay (0-35), default 2 (a typical person, ~78% accuracy)
  rate_scale (float) 1.0 = real time (2 s per window); 4.0 = four times faster
  loop       (bool)  restart the session when it ends
"""
import csv
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32


class DecoderReplay(Node):
    def __init__(self):
        super().__init__("decoder_replay")
        default_csv = os.path.join(get_package_share_directory("eeg_workload_robot"), "data", "decoder_stream.csv")
        self.declare_parameter("csv_path", default_csv)
        self.declare_parameter("subject", 2)
        self.declare_parameter("rate_scale", 1.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("step_s", 2.0)

        path = self.get_parameter("csv_path").value
        subject = int(self.get_parameter("subject").value)
        with open(path, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if int(r["subject"]) == subject]
        if not self.rows:
            raise RuntimeError(f"no windows for subject {subject} in {path}")
        self.rows.sort(key=lambda r: int(r["window"]))
        self.i = 0

        self.pub_p = self.create_publisher(Float32, "/workload/probability", 10)
        self.pub_y = self.create_publisher(Int32, "/workload/label", 10)
        self.pub_done = self.create_publisher(Bool, "/workload/session_done", 10)
        period = float(self.get_parameter("step_s").value) / float(self.get_parameter("rate_scale").value)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f"Replaying subject {subject}: {len(self.rows)} windows, one every {period:.2f} s "
                               "(first 60 s rest, then 60 s mental arithmetic)")

    def tick(self):
        if self.i >= len(self.rows):
            self.pub_done.publish(Bool(data=True))
            if self.get_parameter("loop").value:
                self.i = 0
                self.get_logger().info("Session finished, looping")
            else:
                self.get_logger().info("Session finished")
                self.timer.cancel()
            return
        r = self.rows[self.i]
        self.pub_y.publish(Int32(data=int(r["label"])))           # label first, so controllers can score
        self.pub_p.publish(Float32(data=float(r["p_high_workload"])))
        state = "ARITHMETIC" if r["label"] == "1" else "rest"
        self.get_logger().info(f"t={float(r['t_s']):5.1f} s  true={state:10s}  p(high workload)={float(r['p_high_workload']):.2f}")
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = DecoderReplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
