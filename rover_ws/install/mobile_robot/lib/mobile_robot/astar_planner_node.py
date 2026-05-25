#!/usr/bin/env python3
"""
astar_planner_node.py
======================
ROS2 node that wraps A* algorithm and drives the rover
along the computed path.

What this file does
───────────────────
1. Loads the pre-built map matrices (.npy files) that your map_processing
   script produces: newMap.npy  and  newMapLayers.npy
2. Accepts a goal (waypoint name or raw world coords) via ROS2 topic/service
3. Runs GlobalPath.aStar() — your exact algorithm, zero changes
4. Converts the grid-cell path → world (x,y) metres
5. Publishes the path to /waypoints so rover_controller.py follows it
6. Also publishes a nav_msgs/Path for RViz visualisation

Dependencies
────────────
    pip install numpy plyfile scipy   (already needed by your A* code)
    ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-visualization-msgs

File layout expected
────────────────────
    mobile_robot/
    ├── mobile_robot/
    │   ├── astar_planner_node.py
    │   ├── rover_controller.py
    │   ├── global_path.py
    │   └── map_processing.py
    └── maps/
        ├── newMap.npy
        ├── newMapLayers.npy
        └── newMapPoints.npy

How to trigger planning
────────────────────────
    # By name (S4, W1...W9):
    ros2 service call /plan_path std_srvs/srv/Trigger {}
    # (set the goal name first via parameter or topic — see below)

    # By world coords (e.g. from RViz 2D Nav Goal):
    ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
        "{pose: {position: {x: 7.58, y: 10.874}}}" --once
"""

import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from rcl_interfaces.msg import ParameterDescriptor

# ── Import A* code ────────────────────────────────────────────────────────
# Place your A* class in  mobile_robot/global_path.py  and import it here.
# The class needs no changes at all — we use it exactly as-is.
try:
    from mobile_robot.global_path import GlobalPath, checkCoordinates
except ImportError:
    # Fallback for running the file directly (not via ros2 run)
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from global_path import GlobalPath, checkCoordinates


# ══════════════════════════════════════════════════════════════════════════════
#  COORDINATE CONVERSION CONSTANTS
#  ─────────────────────────────────
#  Your map_processing.py builds the matrix with:
#      xy = (round((x - xMin) * mul),  round((y - yMin) * mul))
#      mul = 6   ->  1 matrix cell = 1/6 m = 0.1667 m
#
#  To reverse:  world_x = grid_row / mul + xMin
#  xMin and yMin are recovered from the saved newMapPoints.npy anchor.
# ══════════════════════════════════════════════════════════════════════════════
MAP_MUL = 6      # must match `mul` in map_processing.py

# Named waypoints in WORLD metres — copied from map_processing.py main()
NAMED_WAYPOINTS_WORLD = {
    "S4": [11.549, 17.228],
    "W1": [-2.259, 13.485],
    "W2": [10.337,  6.433],
    "W3": [ 7.58,  10.874],
    "W4": [ 6.71,  19.515],
    "W5": [ 4.851, 25.405],
    "W6": [ 0.211,  9.454],
    "W7": [ 0.83,  19.867],
    "W8": [ 9.082, 23.409],
    "W9": [ 1.49,  14.13 ],
}

# GlobalPath constructor args — same values as your test() call
ROVER_SIZE          = 8
DIST_BETWEEN_FIELDS = 1.0 / MAP_MUL   # 0.1667 m
ROVER_MAX_SLOPE     = 1.0
SLOPE_THRESHOLD_INC = 0.05


class AStarPlannerNode(Node):

    def __init__(self):
        super().__init__('astar_planner')

        # ── ROS2 parameters ────────────────────────────────────────────
        self.declare_parameter(
            'maps_dir',
            os.path.join(os.path.expanduser('~'), 'maps'),
            ParameterDescriptor(description='Folder containing .npy map files'))
        self.declare_parameter('map_name',                'newMap')
        self.declare_parameter('initial_slope_threshold',  0.05)
        self.declare_parameter('search_depth',             4)
        self.declare_parameter('waypoint_accept_radius',   0.35)  # metres
        self.declare_parameter('downsample_dist',          0.5)   # metres

        maps_dir           = self.get_parameter('maps_dir').value
        map_name           = self.get_parameter('map_name').value
        self.initial_slope = self.get_parameter('initial_slope_threshold').value
        self.search_depth  = self.get_parameter('search_depth').value

        # ── State ──────────────────────────────────────────────────────
        self.map_matrix        = None
        self.map_matrix_layers = None
        self.global_path       = None
        self.xMin = 0.0
        self.yMin = 0.0
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0
        self._last_goal_name: str | None = None

        # ── Load map matrices ──────────────────────────────────────────
        self._load_map(maps_dir, map_name)

        # ── ROS interfaces ─────────────────────────────────────────────
        self.wp_pub     = self.create_publisher(PoseArray,   '/waypoints',    10)
        self.path_pub   = self.create_publisher(Path,        '/astar_path',   10)
        self.marker_pub = self.create_publisher(MarkerArray, '/path_markers', 10)

        self.create_subscription(Odometry,    '/odom',      self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.create_service(Trigger, '/plan_path', self._plan_srv)

        self.get_logger().info("A* Planner node ready.")
        self._log_status(maps_dir, map_name)

    # ══════════════════════════════════════════════════════════════════
    #  MAP LOADING
    # ══════════════════════════════════════════════════════════════════
    def _load_map(self, maps_dir: str, map_name: str):
        map_path    = os.path.join(maps_dir, f'{map_name}.npy')
        layers_path = os.path.join(maps_dir, f'{map_name}Layers.npy')
        points_path = os.path.join(maps_dir, f'{map_name}Points.npy')

        if not os.path.exists(map_path):
            self.get_logger().error(
                f"Map not found: {map_path}\n"
                "Run map_processing.py first to generate the .npy files.")
            return

        self.map_matrix = np.load(map_path)
        self.get_logger().info(
            f"Loaded map: {map_path}  shape={self.map_matrix.shape}")

        if os.path.exists(layers_path):
            self.map_matrix_layers = np.load(layers_path)
        else:
            self.map_matrix_layers = np.zeros_like(self.map_matrix)
            self.get_logger().warn(
                f"Layers file not found ({layers_path}), using zeros.")

        # ── Recover xMin / yMin ────────────────────────────────────────
        # map_processing transforms points in-place before saving them:
        #   point[0] = round((point[0] - xMin) * mul)  <- this overwrites world coords!
        # So newMapPoints.npy stores GRID coordinates, not world coords.
        # We know S4's world position, so:
        #   grid_row = round((world_x - xMin) * mul)
        #   xMin = world_x - grid_row / mul
        if os.path.exists(points_path):
            saved_pts = np.load(points_path)   # shape (N, 2), in grid space
            s4_world  = NAMED_WAYPOINTS_WORLD['S4']
            s4_grid   = saved_pts[0]           # S4 is first in main()
            self.xMin = s4_world[0] - float(s4_grid[0]) / MAP_MUL
            self.yMin = s4_world[1] - float(s4_grid[1]) / MAP_MUL
            self.get_logger().info(
                f"Coordinate origin: xMin={self.xMin:.4f}  yMin={self.yMin:.4f}")
        else:
            self.get_logger().warn(
                f"{points_path} not found — world<->grid conversion uses xMin=yMin=0.\n"
                "This means goals given in world metres will map to wrong cells!\n"
                "Fix: make sure map_processing.py saves newMapPoints.npy.")

        self.global_path = GlobalPath(
            self.map_matrix,
            self.map_matrix_layers,
            ROVER_SIZE,
            DIST_BETWEEN_FIELDS,
            ROVER_MAX_SLOPE,
            SLOPE_THRESHOLD_INC,
        )

    # ══════════════════════════════════════════════════════════════════
    #  COORDINATE HELPERS
    # ══════════════════════════════════════════════════════════════════
    def _world_to_grid(self, wx: float, wy: float) -> list:
        """World metres -> grid [row, col].  Mirrors map_processing.py."""
        row = round((wx - self.xMin) * MAP_MUL)
        col = round((wy - self.yMin) * MAP_MUL)
        return [row, col]

    def _grid_to_world(self, row: int, col: int) -> tuple:
        """Grid [row, col] -> world (x, y) metres."""
        wx = float(row) / MAP_MUL + self.xMin
        wy = float(col) / MAP_MUL + self.yMin
        return wx, wy

    # ══════════════════════════════════════════════════════════════════
    #  ODOMETRY
    # ══════════════════════════════════════════════════════════════════
    def _odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    # ══════════════════════════════════════════════════════════════════
    #  GOAL INPUTS
    # ══════════════════════════════════════════════════════════════════
    def _goal_cb(self, msg: PoseStamped):
        """Receive goal in world metres (RViz 2D Nav Goal or manual publish)."""
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.get_logger().info(f"Goal received: world ({gx:.3f}, {gy:.3f})")
        self._run_and_publish((self.robot_x, self.robot_y), (gx, gy))

    def _plan_srv(self, request, response):
        """
        /plan_path  (std_srvs/Trigger)
        Plans from current odometry position to the last named goal,
        defaulting to 'W3' if none set yet.
        To change the target, publish to /goal_pose or call
        plan_to_named_waypoint() programmatically.
        """
        name = self._last_goal_name or 'W3'
        ok = self.plan_to_named_waypoint(name)
        response.success = ok
        response.message = f"Planned to '{name}'." if ok else "Planning failed."
        return response

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC: plan to a named waypoint
    # ══════════════════════════════════════════════════════════════════
    def plan_to_named_waypoint(self, name: str) -> bool:
        """
        Called programmatically or via the Trigger service.

        Example from another node:
            self.planner.plan_to_named_waypoint('W3')
        """
        if name not in NAMED_WAYPOINTS_WORLD:
            self.get_logger().error(
                f"Unknown waypoint '{name}'. "
                f"Valid: {list(NAMED_WAYPOINTS_WORLD.keys())}")
            return False
        self._last_goal_name = name
        goal = tuple(NAMED_WAYPOINTS_WORLD[name])
        self.get_logger().info(
            f"Planning to '{name}' = world ({goal[0]:.3f}, {goal[1]:.3f})")
        return self._run_and_publish((self.robot_x, self.robot_y), goal)

    # ══════════════════════════════════════════════════════════════════
    #  CORE PLANNING PIPELINE
    # ══════════════════════════════════════════════════════════════════
    def _run_and_publish(self,
                         start_world: tuple,
                         goal_world:  tuple) -> bool:

        if self.global_path is None:
            self.get_logger().error("Map not loaded — cannot plan.")
            return False

        start_grid = self._world_to_grid(*start_world)
        goal_grid  = self._world_to_grid(*goal_world)
        rd = self.global_path.roverDist

        # Validate both cells
        for label, g in [('Start', start_grid), ('Goal', goal_grid)]:
            if not checkCoordinates(g[0], g[1], self.global_path.mapMatrix, rd):
                self.get_logger().error(
                    f"{label} grid cell {g} is out-of-bounds or in obstacle.\n"
                    f"  world=({start_world if label=='Start' else goal_world})\n"
                    f"  map shape={self.global_path.mapMatrix.shape}")
                return False

        self.get_logger().info(
            f"A* start: world{start_world} -> grid{start_grid}\n"
            f"A* goal:  world{goal_world}  -> grid{goal_grid}")

        # ── Run YOUR A* (no changes to GlobalPath) ─────────────────────
        path_grid = self.global_path.aStar(
            start_grid,
            goal_grid,
            searchDepth=self.search_depth,
            initialSlopeThreshold=self.initial_slope,
        )

        if not path_grid:
            self.get_logger().warn(
                "A* found no path. Possible causes:\n"
                "  - Start or goal buried under inf values (scan gap)\n"
                "  - Max slope too tight for this section of terrain\n"
                "  - Rover size too large relative to grid cell count\n"
                "Try: raise roverMaxSlope, reduce roverSize, or pick nearby goal.")
            return False

        # ── Grid path -> world coords ───────────────────────────────────
        path_world = [self._grid_to_world(r, c) for r, c in path_grid]

        # ── Downsample: keep 1 waypoint per ~0.5 m ─────────────────────
        # Your A* returns ~1 cell/step (0.167 m).  Sending every cell to
        # the controller causes micro-jitter.  0.5 m spacing is smooth.
        min_d = self.get_parameter('downsample_dist').value
        path_world = _downsample(path_world, min_d)

        self.get_logger().info(
            f"Path: {len(path_grid)} grid cells -> "
            f"{len(path_world)} waypoints after downsampling.")

        stamp = self.get_clock().now().to_msg()
        self._pub_pose_array(path_world, stamp)
        self._pub_nav_path(path_world, stamp)
        self._pub_markers(path_world, goal_world, stamp)
        return True

    # ══════════════════════════════════════════════════════════════════
    #  PUBLISHERS
    # ══════════════════════════════════════════════════════════════════
    def _pub_pose_array(self, path_world, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        for wx, wy in path_world:
            p = Pose()
            p.position.x = float(wx)
            p.position.y = float(wy)
            p.orientation.w = 1.0
            msg.poses.append(p)
        self.wp_pub.publish(msg)

    def _pub_nav_path(self, path_world, stamp):
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        for wx, wy in path_world:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _pub_markers(self, path_world, goal_world, stamp):
        arr = MarkerArray()
        # Clear old markers
        clr = Marker(); clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        # Waypoint spheres
        for i, (wx, wy) in enumerate(path_world):
            m = Marker()
            m.header.stamp = stamp; m.header.frame_id = 'odom'
            m.ns = 'waypoints'; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(wx); m.pose.position.y = float(wy)
            m.pose.position.z = 0.15; m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 0.0; m.color.g = 0.4; m.color.b = 1.0; m.color.a = 0.8
            arr.markers.append(m)
        # Goal cylinder
        g = Marker()
        g.header.stamp = stamp; g.header.frame_id = 'odom'
        g.ns = 'goal'; g.id = 9999
        g.type = Marker.CYLINDER; g.action = Marker.ADD
        g.pose.position.x = float(goal_world[0]); g.pose.position.y = float(goal_world[1])
        g.pose.position.z = 0.5; g.pose.orientation.w = 1.0
        g.scale.x = g.scale.y = 0.4; g.scale.z = 1.0
        g.color.r = 0.0; g.color.g = 1.0; g.color.b = 0.0; g.color.a = 0.9
        arr.markers.append(g)
        self.marker_pub.publish(arr)

    # ══════════════════════════════════════════════════════════════════
    def _log_status(self, maps_dir, map_name):
        shape = self.map_matrix.shape if self.map_matrix is not None else 'NOT LOADED'
        self.get_logger().info(
            f"Map shape: {shape} | "
            f"xMin={self.xMin:.3f} yMin={self.yMin:.3f} | "
            f"cell size={1/MAP_MUL:.4f}m")
        self.get_logger().info(
            f"Known waypoints: {list(NAMED_WAYPOINTS_WORLD.keys())}")
        self.get_logger().info(
            "Send a goal:  ros2 topic pub /goal_pose "
            'geometry_msgs/msg/PoseStamped "{pose: {position: {x: 7.58, y: 10.874}}}" --once')


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS (module-level)
# ══════════════════════════════════════════════════════════════════════════════
def _downsample(path: list, min_dist: float) -> list:
    """
    Keep waypoints that are at least min_dist metres apart.
    Always keeps first and last.

    Why: A* returns one cell per step (~0.167 m). The rover controller
    doesn't need to aim at every cell — that causes micro-stop-and-turn
    behaviour. Spacing at ~0.5 m gives smooth forward motion.
    """
    if len(path) <= 2:
        return path
    kept = [path[0]]
    for pt in path[1:-1]:
        if math.hypot(pt[0] - kept[-1][0], pt[1] - kept[-1][1]) >= min_dist:
            kept.append(pt)
    kept.append(path[-1])
    return kept


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()