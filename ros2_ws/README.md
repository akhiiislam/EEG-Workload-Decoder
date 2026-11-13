# ROS 2 demo: a robot that adapts its speed to the EEG workload decoder

A simulated robot slows down when the EEG decoder says the person is overloaded, and acts only when the decoder is confident.

- The decoder outputs come from the main project's **calibrated leave-one-subject-out model** (75.2% balanced accuracy on unseen people, ECE 0.024).
- They are exported to `src/eeg_workload_robot/data/decoder_stream.csv` and replayed in real time, one window every 2 s.
- This is a simulation for demonstration. It does **not** read live EEG or drive a physical robot.

## What you will see

The `compare` launch opens **turtlesim** with two robots driving in circles.

| Turtle | Policy | Rule |
|---|---|---|
| Left (turtle1) | **naive** | Slow if p ≥ 0.5, otherwise normal. It reacts to every decoder output. |
| Right (turtle2) | **confidence-aware + smoothing** | Averages the last 3 outputs (6 s). Slow if the mean ≥ 0.8, normal if ≤ 0.2, otherwise cautious. |

- **Pen colours:** 🟢 normal speed, 🟠 cautious, 🔴 slow.
- **The session:** the replayed person rests for 60 s, then does mental arithmetic for 60 s.

| Robot (all 36 people) | Full speed while overloaded | Slowed for no reason | Speed changes / min |
|---|---|---|---|
| naive | 24.8% | 24.8% | 7.4 |
| confidence-aware | 6.1% | 6.1% | 11.4 |
| confidence-aware + smoothing | **1.7%** | **1.3%** | **4.8** |

`test/test_policy.py` checks that the ROS policies reproduce these numbers exactly.

## Run it on Ubuntu

### 1. Check your Ubuntu and ROS 2 versions
```bash
lsb_release -a
printenv ROS_DISTRO
```
- Ubuntu **22.04** uses ROS 2 **Humble**, and Ubuntu **24.04** uses **Jazzy**.
- If `ROS_DISTRO` prints nothing, install ROS 2 **Desktop** (it includes turtlesim and rqt) using the official guide: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html. For 24.04, replace `humble` with `jazzy` in the link.
- Also install the build tool:
```bash
sudo apt install python3-colcon-common-extensions
```

### 2. Get the code
```bash
cd ~
git clone https://github.com/akhiiislam/EEG-Workload-Decoder.git
cd EEG-Workload-Decoder/ros2_ws
```
If you prefer, copy the folder from the Windows drive. In Ubuntu's file manager it appears under *Other Locations*, at `D:\PhD\EEG_Workload_Decoder\ros2_ws`.

### 3. Build
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 4. Run the side-by-side demo
```bash
ros2 launch eeg_workload_robot compare.launch.py subject:=2
```
- `subject:=0` to `35` replays a different person. Person 2 is typical, at about 78% accuracy.
- `rate_scale:=4.0` plays 4× faster. The default is real time: about 2 minutes per person.
- The terminal prints each decision. A line ending in `<-- full speed while overloaded!` marks an unsafe moment.
- At the end, each robot prints a **SESSION SUMMARY**.
- A CSV log of every decision is saved in `~/eeg_robot_logs/`.

### 5. Live plot (optional, in a second terminal)
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 run rqt_plot rqt_plot /workload/probability/data /turtle1/speed_scale/data /turtle2/speed_scale/data
```

### 6. Record a video for sharing
- **Ubuntu 22.04 / 24.04:** press **Print Screen**, choose the **video** (screencast) option, select the turtlesim window, and press Record. Press it again to stop. The video is saved in `~/Videos/Screencasts`.
- **Other tool:** `sudo apt install simplescreenrecorder` also works well.

### 7. Run the tests
```bash
cd ~/EEG-Workload-Decoder/ros2_ws/src/eeg_workload_robot
python3 -m pytest -q test/
```

## One robot, or a Gazebo robot

```bash
# one turtle with a chosen policy (naive | aware | smooth)
ros2 launch eeg_workload_robot single.launch.py policy:=smooth subject:=2

# any robot that listens to a Twist topic, e.g. a TurtleBot3 already running in Gazebo
ros2 launch eeg_workload_robot single.launch.py use_turtlesim:=false cmd_vel_topic:=/cmd_vel base_linear:=0.15 base_angular:=0.3
```

## Topics

| Topic | Type | Meaning |
|---|---|---|
| `/workload/probability` | `std_msgs/Float32` | Decoder output P(high workload), every 2 s |
| `/workload/label` | `std_msgs/Int32` | True condition (0 rest, 1 arithmetic), used only to score the robot |
| `/workload/session_done` | `std_msgs/Bool` | End of the replayed session |
| `/<turtle>/cmd_vel` | `geometry_msgs/Twist` | Robot velocity, 10 Hz |
| `/<turtle>/speed_scale` | `std_msgs/Float32` | 1.0 normal, 0.7 cautious, 0.4 slow |
| `/<turtle>/mode` | `std_msgs/String` | normal / cautious / slow |

## Regenerate the decoder stream (on the machine with the EEG data)
```bash
python export_decoder_stream.py
```
