"""
gazebo_mars_launch.py
======================
Updated launch file for ERC 2026 Mars Yard simulation.

Launches:
  1. Gazebo with mars_yard.world
  2. robot_state_publisher (reads robot.xacro)
  3. spawn_entity (spawns rover at start position)
  4. rover_controller  (teleop + auto waypoint follower)
  5. astar_planner_node (A* path planner)

Usage:
  ros2 launch mobile_robot gazebo_mars_launch.py
  ros2 launch mobile_robot gazebo_mars_launch.py mode:=auto
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # ── Package / file paths ──────────────────────────────────────────
    pkg = 'mobile_robot'
    pkg_dir = get_package_share_directory(pkg)

    robot_xacro_name = 'differential_drive_robot'
    path_xacro  = os.path.join(pkg_dir, 'model', 'robot.xacro')
    path_world  = os.path.join(pkg_dir, 'model', 'mars_yard.world')

    robot_description_xml = xacro.process_file(path_xacro).toxml()

    # ── Launch arguments ─────────────────────────────────────────────
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='teleop',
        description="rover_controller mode: 'teleop' or 'auto'"
    )
    mode = LaunchConfiguration('mode')

    # ── 1. Gazebo ─────────────────────────────────────────────────────
    gazebo_launch_src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('gazebo_ros'),
                     'launch', 'gazebo.launch.py')
    )
    gazebo = IncludeLaunchDescription(
        gazebo_launch_src,
        launch_arguments={'world': path_world, 'verbose': 'false'}.items()
    )

    # ── 2. Robot State Publisher ──────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_xml,
            'use_sim_time': True
        }]
    )

    # ── 3. Spawn rover in Gazebo ──────────────────────────────────────
    # Spawned at (0, 0) — the "start" position in the Mars Yard
    spawn_rover = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_xacro_name,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.3',    # slight Z offset so wheels don't clip ground
            '-Y', '0.0',    # initial yaw (facing +X)
        ],
        output='screen'
    )

    # ── 4. Rover controller ───────────────────────────────────────────
    rover_controller = Node(
        package='mobile_robot',
        executable='rover_controller.py',
        name='rover_controller',
        output='screen',
        parameters=[{'mode': mode}]
    )

    # ── 5. A* Planner ────────────────────────────────────────────────
    astar_planner = Node(
        package='mobile_robot',
        executable='astar_planner_node.py',
        name='astar_planner',
        output='screen'
    )

    # ── Assemble LaunchDescription ────────────────────────────────────
    ld = LaunchDescription()
    ld.add_action(mode_arg)
    ld.add_action(gazebo)
    ld.add_action(robot_state_publisher)
    ld.add_action(spawn_rover)
    ld.add_action(rover_controller)
    ld.add_action(astar_planner)
    return ld
