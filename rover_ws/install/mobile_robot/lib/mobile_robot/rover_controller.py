#!/usr/bin/env python3
"""
rover_controller.py
====================
ROS2 Humble — ERC Rover Controller

Replaces the original move.py (which just did forward->turn->stop in a
fixed time loop).  This node does two things:

  TELEOP mode (default):
    WASD / arrow keys drive the rover in real time.
    Emergency stop, speed adjustment, smooth decay.

  AUTO mode:
    Follows a list of (x, y) waypoints published by astar_planner_node.py
    on the /waypoints topic, using a simple proportional heading controller.
    Press M to toggle between modes at any time.

Topics
──────
  Publishes  :  /cmd_vel          (geometry_msgs/Twist)
  Subscribes :  /odom             (nav_msgs/Odometry)       — AUTO mode
  Subscribes :  /waypoints        (geometry_msgs/PoseArray) — AUTO mode

Key bindings (TELEOP)
──────────────────────
  W / Up     Forward
  S / Down   Backward
  A / Left   Turn left (CCW)
  D / Right  Turn right (CW)
  Space      Emergency stop
  M          Toggle AUTO / TELEOP
  + / -      Speed up / down
  Esc / ^C   Quit
"""

import sys
import tty
import termios
import threading
import select
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry


# ──────────────────────────────────────────────────────────────────────────────
#  TUNING  (change these to match your rover's physical behaviour)
# ──────────────────────────────────────────────────────────────────────────────
LINEAR_SPEED_DEFAULT  = 0.5    # m/s
ANGULAR_SPEED_DEFAULT = 0.8    # rad/s
SPEED_STEP            = 0.05   # per keypress
MAX_LINEAR            = 1.5    # hard ceiling m/s
MAX_ANGULAR           = 2.0    # hard ceiling rad/s

# AUTO mode
KP_LINEAR             = 0.6    # proportional gain for distance
KP_ANGULAR            = 1.5    # proportional gain for heading error
HEADING_ONLY_THRESH   = 0.4    # rad  (~23°) — rotate in place above this
WAYPOINT_ACCEPT_DIST  = 0.35   # m    — distance to count a waypoint as reached

# Teleop decay: if no key is pressed for this many seconds, slow down to zero.
# This prevents the rover from running away if the terminal loses focus.
TELEOP_DECAY_TIMEOUT  = 0.3    # seconds


class RoverController(Node):

    def __init__(self):
        super().__init__('rover_controller')

        # ── ROS2 parameter: start in teleop or auto ────────────────────
        self.declare_parameter('mode', 'teleop')
        mode_val = self.get_parameter('mode').get_parameter_value().string_value
        self.auto_mode = (mode_val.lower() == 'auto')

        # ── Speeds ────────────────────────────────────────────────────
        self.linear_speed  = LINEAR_SPEED_DEFAULT
        self.angular_speed = ANGULAR_SPEED_DEFAULT

        # ── Teleop state ──────────────────────────────────────────────
        self.teleop_twist   = Twist()
        self._last_key_time = self.get_clock().now()

        # ── Odometry state ────────────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # ── Waypoint queue (AUTO) ─────────────────────────────────────
        self.waypoints: list[tuple[float, float]] = []
        self.wp_index  = 0

        # ── ROS interfaces ────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(Odometry,  '/odom',      self._odom_cb, 10)
        self.create_subscription(PoseArray, '/waypoints', self._wp_cb,   10)

        # Control loop at 20 Hz
        self.create_timer(0.05, self._control_loop)

        # Keyboard thread (daemon — dies with the main process)
        threading.Thread(target=self._keyboard_loop, daemon=True).start()

        mode_str = "AUTO" if self.auto_mode else "TELEOP"
        self.get_logger().info(f"Rover controller started in {mode_str} mode.")
        _print_help()

    # ══════════════════════════════════════════════════════════════════
    #  KEYBOARD
    # ══════════════════════════════════════════════════════════════════
    def _keyboard_loop(self):
        settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info("Keyboard listener active.")
        try:
            while rclpy.ok():
                key = _get_key(settings)
                if key:
                    self._handle_key(key)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    def _handle_key(self, key: str):
        # ── Global keys (work in any mode) ────────────────────────────
        if key in ('\x1b', '\x03'):          # Esc or Ctrl-C
            self.get_logger().info("Shutting down.")
            rclpy.shutdown()
            return

        if key == ' ':                        # Emergency stop
            self.teleop_twist = Twist()
            self.waypoints = []
            self.get_logger().info("!! EMERGENCY STOP !!")
            return

        if key == 'm':                        # Toggle mode
            self.auto_mode = not self.auto_mode
            self.teleop_twist = Twist()       # clear motion on switch
            mode_str = "AUTO" if self.auto_mode else "TELEOP"
            self.get_logger().info(f"Mode -> {mode_str}")
            return

        if key == '+':
            self.linear_speed  = min(self.linear_speed  + SPEED_STEP, MAX_LINEAR)
            self.angular_speed = min(self.angular_speed + SPEED_STEP, MAX_ANGULAR)
            self.get_logger().info(
                f"Speed: lin={self.linear_speed:.2f}  ang={self.angular_speed:.2f}")
            return

        if key == '-':
            self.linear_speed  = max(self.linear_speed  - SPEED_STEP, 0.05)
            self.angular_speed = max(self.angular_speed - SPEED_STEP, 0.1)
            self.get_logger().info(
                f"Speed: lin={self.linear_speed:.2f}  ang={self.angular_speed:.2f}")
            return

        # ── Teleop movement keys ───────────────────────────────────────
        if self.auto_mode:
            return  # movement keys do nothing in AUTO mode

        twist = Twist()
        if   key in ('w', '\x1b[A'):   twist.linear.x  =  self.linear_speed
        elif key in ('s', '\x1b[B'):   twist.linear.x  = -self.linear_speed
        elif key in ('a', '\x1b[D'):   twist.angular.z =  self.angular_speed
        elif key in ('d', '\x1b[C'):   twist.angular.z = -self.angular_speed

        self.teleop_twist = twist
        self._last_key_time = self.get_clock().now()

    # ══════════════════════════════════════════════════════════════════
    #  SUBSCRIBERS
    # ══════════════════════════════════════════════════════════════════
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def _wp_cb(self, msg: PoseArray):
        """Receive waypoint list from astar_planner_node."""
        self.waypoints = [(p.position.x, p.position.y) for p in msg.poses]
        self.wp_index  = 0
        self.auto_mode = True   # auto-switch to AUTO when path arrives
        self.get_logger().info(
            f"Received {len(self.waypoints)} waypoints — AUTO mode ON.")

    # ══════════════════════════════════════════════════════════════════
    #  AUTONOMOUS CONTROLLER
    # ══════════════════════════════════════════════════════════════════
    def _auto_twist(self) -> Twist:
        """
        Proportional heading + distance controller.

        Logic:
          1. Compute vector from robot to current waypoint.
          2. Compute heading error = desired_yaw - robot_yaw.
          3. If |error| > HEADING_ONLY_THRESH → rotate in place (no linear).
          4. Otherwise → drive forward + correct heading simultaneously.
          5. When within WAYPOINT_ACCEPT_DIST → advance to next waypoint.
        """
        twist = Twist()

        if self.wp_index >= len(self.waypoints):
            if self.waypoints:
                self.get_logger().info(
                    "All waypoints reached.", throttle_duration_sec=5.0)
            return twist   # stop

        wx, wy = self.waypoints[self.wp_index]
        dx = wx - self.robot_x
        dy = wy - self.robot_y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_ACCEPT_DIST:
            self.get_logger().info(
                f"Waypoint {self.wp_index}/{len(self.waypoints)-1} "
                f"reached: ({wx:.2f}, {wy:.2f})")
            self.wp_index += 1
            return twist

        desired_yaw  = math.atan2(dy, dx)
        heading_err  = desired_yaw - self.robot_yaw

        # Normalize to [-pi, pi]
        while heading_err >  math.pi: heading_err -= 2 * math.pi
        while heading_err < -math.pi: heading_err += 2 * math.pi

        ang_cmd = float(max(min(KP_ANGULAR * heading_err, MAX_ANGULAR), -MAX_ANGULAR))

        if abs(heading_err) > HEADING_ONLY_THRESH:
            # Large angle error — rotate in place
            twist.angular.z = ang_cmd
            twist.linear.x  = 0.0
        else:
            # Drive forward + fine-tune heading
            twist.linear.x  = float(min(KP_LINEAR * dist, self.linear_speed))
            twist.angular.z = ang_cmd

        return twist

    # ══════════════════════════════════════════════════════════════════
    #  CONTROL LOOP  (20 Hz)
    # ══════════════════════════════════════════════════════════════════
    def _control_loop(self):
        if self.auto_mode:
            twist = self._auto_twist()
        else:
            # Apply decay: if no key pressed recently, zero out
            now = self.get_clock().now()
            dt  = (now - self._last_key_time).nanoseconds * 1e-9
            if dt > TELEOP_DECAY_TIMEOUT:
                self.teleop_twist = Twist()
            twist = self.teleop_twist

        self.cmd_pub.publish(twist)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_key(settings) -> str:
    """Read one key with 0.1 s timeout. Returns '' on timeout."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def _print_help():
    print("""
╔══════════════════════════════════════════════════════╗
║          ERC Rover Controller  (ROS2 Humble)         ║
╠══════════════════════════════════════════════════════╣
║  W / ↑   Forward          S / ↓   Backward           ║
║  A / ←   Turn Left        D / →   Turn Right         ║
║  Space   Emergency Stop                              ║
║  M       Toggle AUTO / TELEOP                        ║
║  + / -   Speed up / down                             ║
║  Esc     Quit                                        ║
╠══════════════════════════════════════════════════════╣
║  AUTO mode activates automatically when A* publishes ║
║  waypoints on /waypoints.                            ║
╚══════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = RoverController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always send a stop before dying
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()