"""Bring up Gazebo, the robot, controllers and the perception/execution stack.

Usage:
  ros2 launch ag_bringup sim.launch.py                        # realtime GUI
  ros2 launch ag_bringup sim.launch.py headless:=true fast:=true   # training
"""

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_ag_gazebo = get_package_share_directory("ag_gazebo")
    pkg_ag_description = get_package_share_directory("ag_description")
    pkg_ag_perception = get_package_share_directory("ag_perception")
    pkg_ag_execution = get_package_share_directory("ag_execution")
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")

    world_realtime = PathJoinSubstitution([pkg_ag_gazebo, "worlds", "tabletop.world"])
    world_fast = PathJoinSubstitution([pkg_ag_gazebo, "worlds", "tabletop_fast.world"])

    controller_config = os.path.join(
        pkg_ag_description, "config", "ag_arm_controllers.yaml"
    )
    urdf_file = os.path.join(pkg_ag_description, "urdf", "ag_arm.urdf")
    with open(urdf_file, "r") as f:
        robot_description = f.read()
    # gazebo_ros2_control 插件会自己创建 controller_manager，
    # 需要把控制器配置文件的安装路径写进 URDF 模板占位符。
    robot_description = robot_description.replace(
        "__AG_CONTROLLERS_YAML__", controller_config
    )
    # gazebo_ros2_control 插件会把整段 URDF 作为“参数覆盖规则”注入控制器管理器，
    # 该解析基于 YAML：XML 注释、声明和换行都会破坏它（社区已知问题；xacro 生成的
    # URDF 恰好没有注释且是单行，所以官方示例能跑）。这里清理成纯标签文本。
    robot_description = re.sub(r"<!--.*?-->", " ", robot_description, flags=re.S)
    robot_description = re.sub(r"<\?xml[^>]*\?>", "", robot_description)
    robot_description = " ".join(robot_description.split())
    perception_config = os.path.join(
        pkg_ag_perception, "config", "perception.yaml"
    )
    execution_config = os.path.join(
        pkg_ag_execution, "config", "execution.yaml"
    )

    gazebo_source = os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")

    gazebo_realtime = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_source),
        launch_arguments={
            "world": world_realtime,
            "gui": LaunchConfiguration("gui"),
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("fast")),
    )
    gazebo_fast = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_source),
        launch_arguments={"world": world_fast, "gui": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("fast")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("fast", default_value="false"),
            gazebo_realtime,
            gazebo_fast,
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {"robot_description": robot_description, "use_sim_time": True}
                ],
                output="screen",
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=[
                    "-topic", "robot_description",
                    "-entity", "ag_arm",
                    "-x", "0.0", "-y", "0.0", "-z", "0.78",
                    # WSL 冷启动时 gzserver 加载世界可能要 30~60 s；30 s 会在
                    # 世界刚加载完的前一刻放弃，导致机械臂根本没生成。
                    "-timeout", "180.0",
                ],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["arm_controller"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["gripper_controller"],
                output="screen",
            ),
            Node(
                package="ag_gazebo",
                executable="world_manager",
                output="screen",
            ),
            Node(
                package="ag_perception",
                executable="perception_node",
                parameters=[perception_config],
                output="screen",
            ),
            Node(
                package="ag_execution",
                executable="execution_node",
                parameters=[execution_config, {"urdf_file": urdf_file}],
                output="screen",
            ),
        ]
    )
