import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro

def generate_launch_description():
    #ovo ime mora da bude isto kao u xacro fajlu
    robotXacroName='differential_drive_robot'

    namePackage = 'mobile_robot'

    #relativna putanja do .xacro fajla
    modelFileRelativePath = 'model/robot.xacro'

    #relativna putanja do gazebo .world fajla
    worldFileRelativePath = 'model/empty_world.world'

    #apsolutna putanja do modela
    pathModelFile = os.path.join(get_package_share_directory(namePackage), modelFileRelativePath)

    #apsolutna putanja do world modela
    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRelativePath)

    robotDescription = xacro.process_file(pathModelFile).toxml()

    #ovo je launch fajl iz gazebo_ros paketa
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py'
    ))

    #launch descripton
    gazeboLaunch = IncludeLaunchDescription(gazebo_rosPackageLaunch,
                                            launch_arguments={'world' :pathWorldFile}.items())
    
    # ovde pravimo gazebo_ros Node
    spawnModelNode = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', robotXacroName],
        output='screen'
    )

    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription,
                     'use_sim_time': True}]
    )

    moveControllerNode = Node(
        package='mobile_robot',
        executable='move.py',
        name='move',
        output='screen'
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(gazeboLaunch)

    launchDescriptionObject.add_action(spawnModelNode)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(moveControllerNode)
    return launchDescriptionObject
