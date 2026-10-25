"""ROS node that turns Grasp/Push/MoveView requests into arm motion."""

import math
import threading
import time

import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from gazebo_msgs.srv import GetEntityState, GetWorldProperties
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from ag_interfaces.srv import Grasp, Push, MoveView, Observe
from ag_execution.kinematics import (
    RobotModel, joint_error, look_at_matrix, nearest_joint_branch,
    pose_from_rot_trans,
)
from ag_execution.primitives import (
    grasp_plan, push_plan, view_plan, grid_to_xy, interpolate_joints,
    gripper_width, is_holding,
)


class ExecutionNode(Node):
    def __init__(self):
        super().__init__("execution_node")

        self.declare_parameter("urdf_file", "")
        self.declare_parameter("arm_action", "/arm_controller/follow_joint_trajectory")
        self.declare_parameter("gripper_topic", "/gripper_controller/commands")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("observe_service", "/perception_node/observe")
        self.declare_parameter("base_link", "base_link")
        self.declare_parameter("tool_link", "tool0")
        self.declare_parameter("camera_link", "camera_link")
        self.declare_parameter("home_camera_pos", [0.68, 0.0, 0.62])
        self.declare_parameter("home_camera_look", [0.45, 0.0, 0.02])
        self.declare_parameter("bin_pos", [0.45, -0.34, 0.32])
        # ROS 2 参数不支持“字典列表”，这里用扁平数组：
        # 每个视角 6 个数 = [pos_x, pos_y, pos_z, look_x, look_y, look_z]
        self.declare_parameter("views_flat", [
            0.45, 0.00, 0.72, 0.45, 0.0, 0.02,
            0.45, -0.28, 0.52, 0.45, 0.0, 0.02,
            0.45, 0.28, 0.52, 0.45, 0.0, 0.02,
            0.18, 0.00, 0.60, 0.45, 0.0, 0.02,
        ])
        self.declare_parameter("grid_x_min", 0.20)
        self.declare_parameter("grid_x_max", 0.70)
        self.declare_parameter("grid_y_min", -0.25)
        self.declare_parameter("grid_y_max", 0.25)
        self.declare_parameter("grid_width", 48)
        self.declare_parameter("grid_height", 48)
        self.declare_parameter("min_grasp_height", 0.008)
        self.declare_parameter("gripper_close", [0.034, 0.034])
        self.declare_parameter("gripper_open", [0.0, 0.0])
        self.declare_parameter("gripper_open_gap", 0.070)
        self.declare_parameter("gripper_closed_gap", 0.002)
        self.declare_parameter("held_margin", 0.004)
        self.declare_parameter("held_q_sum", 0.033)
        self.declare_parameter("approach_offset", 0.18)
        self.declare_parameter("grasp_offset", 0.065)
        self.declare_parameter("finger_length", 0.08)
        self.declare_parameter("lift_height", 0.30)
        self.declare_parameter("push_height", 0.075)
        self.declare_parameter("push_length", 0.09)
        self.declare_parameter("view_dwell", 1.2)
        self.declare_parameter("trajectory_steps", 8)
        self.declare_parameter("verify_objects", True)
        # 同样用扁平数组：每个种子组 6 个关节角
        self.declare_parameter("seed_sets_flat", [
            0.0, -0.4, 1.3, -0.5, 0.0, 0.0,
            0.0, -0.8, 1.6, -0.8, 0.0, 0.0,
            0.5, -0.5, 1.2, -0.7, 0.0, 0.0,
            -0.5, -0.5, 1.2, -0.7, 0.0, 0.0,
            0.0, -1.0, 1.0, -0.9, 0.0, 0.0,
        ])

        urdf_file = self.get_parameter("urdf_file").value
        if not urdf_file:
            raise RuntimeError("urdf_file parameter is required")
        with open(urdf_file, "r") as f:
            self.model = RobotModel(f.read(), base_link=self.get_parameter("base_link").value)

        p = self.get_parameter
        self.tool_link = p("tool_link").value
        self.camera_link = p("camera_link").value
        self.base_link = p("base_link").value
        views_flat = [float(v) for v in p("views_flat").value]
        views = [
            {
                "pos": views_flat[i * 6: i * 6 + 3],
                "look": views_flat[i * 6 + 3: i * 6 + 6],
            }
            for i in range(len(views_flat) // 6)
        ]
        self.grid = {
            "x_min": p("grid_x_min").value, "x_max": p("grid_x_max").value,
            "y_min": p("grid_y_min").value, "y_max": p("grid_y_max").value,
            "width": p("grid_width").value, "height": p("grid_height").value,
        }
        self.params = {
            "bin_pos": p("bin_pos").value,
            "views": views,
            "gripper_close": p("gripper_close").value,
            "gripper_open": p("gripper_open").value,
            "gripper_open_gap": p("gripper_open_gap").value,
            "gripper_closed_gap": p("gripper_closed_gap").value,
            "held_margin": p("held_margin").value,
            "approach_offset": p("approach_offset").value,
            "grasp_offset": p("grasp_offset").value,
            "finger_length": p("finger_length").value,
            "lift_height": p("lift_height").value,
            "push_height": p("push_height").value,
            "push_length": p("push_length").value,
            "view_dwell": p("view_dwell").value,
            # WSL 里仿真慢于实时，且 gazebo_ros2_control 的位置环增益有限：
            # 轨迹过快会让关节跟不上，控制器直接以 goal_time_tolerance 中止
            # （日志里 Position Error 0.05~0.16 rad 就是这么来的）。
            "dur_approach": 1.2, "dur_descend": 1.8, "dur_close": 1.4,
            "dur_hold_check": 0.5, "dur_lift": 1.2, "dur_transit": 1.8,
            "dur_open": 0.6, "dur_home": 1.8, "dur_push": 1.0,
            "dur_retreat": 0.8, "dur_view": 1.4,
        }
        self.min_grasp_height = p("min_grasp_height").value
        self.held_q_sum = p("held_q_sum").value
        self.traj_steps = p("trajectory_steps").value
        self.joint_lower, self.joint_upper = self.model.limits
        seeds_flat = [float(v) for v in p("seed_sets_flat").value]
        self.seed_sets = [
            seeds_flat[i * 6: i * 6 + 6] for i in range(len(seeds_flat) // 6)
        ]

        self._joint_positions = {}
        self._hold_width = 0.0
        self._current_q = np.zeros(6)
        self._have_state = False
        self.create_subscription(
            JointState, p("joint_states_topic").value, self._joint_cb,
            qos_profile_sensor_data,
        )

        ready_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._ready_pub = self.create_publisher(Bool, "/ag_execution/ready", ready_qos)
        self._gripper_pub = self.create_publisher(
            Float64MultiArray, p("gripper_topic").value, 10
        )
        self._arm_client = ActionClient(self, FollowJointTrajectory, p("arm_action").value)
        self._observe_cli = self.create_client(Observe, p("observe_service").value)
        self._world_props_cli = self.create_client(
            GetWorldProperties, "/gazebo/get_world_properties"
        )
        self._entity_state_cli = self.create_client(
            GetEntityState, "/gazebo/get_entity_state"
        )
        self.verify_objects = bool(p("verify_objects").value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Blocking manipulation work lives in its own callback group: the
        # executor keeps delivering action results and joint states on other
        # threads while a grasp/push/move callback is waiting.
        self._srv_group = MutuallyExclusiveCallbackGroup()
        self._grasp_srv = self.create_service(
            Grasp, "~/grasp", self._grasp_cb, callback_group=self._srv_group
        )
        self._push_srv = self.create_service(
            Push, "~/push", self._push_cb, callback_group=self._srv_group
        )
        self._view_srv = self.create_service(
            MoveView, "~/move_view", self._view_cb, callback_group=self._srv_group
        )
        self._home_srv = self.create_service(
            Trigger, "~/go_home", self._home_cb, callback_group=self._srv_group
        )

        # Set by main() once the multi threaded executor is spinning; until
        # then the blocking helpers below have to pump the node themselves.
        self._external_spin = False
        self._ready_event = threading.Event()
        self.home_q = None

    # ---------------- state helpers ----------------
    def _joint_cb(self, msg):
        if not all(math.isfinite(v) for v in msg.position):
            # A single bad state message must not poison the IK seeds.
            return
        self._joint_positions = dict(zip(msg.name, msg.position))
        if not self._have_state:
            try:
                q = np.array([self._joint_positions[n] for n in self.model.arm_joints])
                if q.shape == (6,):
                    self._current_q = np.clip(q, self.joint_lower, self.joint_upper)
                    self._have_state = True
            except KeyError:
                pass
        else:
            try:
                q = np.array(
                    [self._joint_positions[n] for n in self.model.arm_joints]
                )
                self._current_q = np.clip(q, self.joint_lower, self.joint_upper)
            except KeyError:
                pass

    def _wait_for_joint_state(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        while not self._have_state and time.monotonic() < deadline and rclpy.ok():
            self._pause()
        if not self._have_state:
            self.get_logger().warn("no joint state; assuming all-zero configuration")

    def _check_gripper_ready(self, timeout=5.0):
        """Fail loudly when the finger joints never show up in /joint_states.

        Those joints only appear when gazebo_ros2_control managed to bind them
        and the gripper controller is active; otherwise every grasp would fail
        the hold check with no obvious explanation.
        """
        needed = ("finger_left", "finger_right")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(name in self._joint_positions for name in needed):
                self.get_logger().info(
                    "gripper joints online: "
                    + ", ".join(f"{n}={self._joint_positions[n]:.4f}" for n in needed)
                )
                return True
            self._pause()
        missing = [n for n in needed if n not in self._joint_positions]
        self.get_logger().error(
            "gripper joints missing from /joint_states: "
            + ", ".join(missing)
            + " -> gripper_controller is not active, every grasp will fail. "
            "Look for 'Skipping joint in the URDF named ...' or "
            "'Failed to activate controller : gripper_controller' in the log."
        )
        return False

    def _sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._pause()

    def _pause(self, timeout_sec=0.02):
        """Yield for a moment while joint states keep being processed.

        Once main() runs a multi threaded executor we only need to sleep; the
        executor's other threads deliver action results and joint states.
        Before that (constructor startup) the node has to be pumped by hand.
        """
        if self._external_spin:
            time.sleep(timeout_sec)
            return
        try:
            rclpy.spin_once(self, timeout_sec=timeout_sec)
        except (ValueError, RuntimeError):
            time.sleep(timeout_sec)

    def _wait_future(self, future, timeout):
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline and rclpy.ok():
            self._pause()
        return future.done()

    def _wait_ready(self, timeout=180.0):
        if self._ready_event.wait(timeout):
            return True
        self.get_logger().warn("execution is not ready yet; rejecting request")
        return False

    # ---------------- startup ----------------
    def startup(self):
        """Wait for the controller, then park the arm in the home pose.

        Called from main() once the multi threaded executor is spinning, so a
        plain wait is enough: the executor's threads deliver the action
        results and joint states.
        """
        self.get_logger().info("waiting for arm controller...")
        while rclpy.ok() and not self._arm_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().info("still waiting for /arm_controller action server ...")
        if not rclpy.ok():
            return
        self._wait_for_joint_state(timeout=15.0)
        self.home_q = self._solve_home()
        self.get_logger().info("moving to home pose...")
        if not self._move_to_joints(self.home_q, 2.0):
            raise RuntimeError("could not reach home pose at startup")
        self._sleep(0.3)  # let /joint_states catch up with the final command
        home_error = joint_error(self._current_q, self.home_q)
        if home_error > 0.1:
            raise RuntimeError(
                f"arm did not reach the home pose (max joint error "
                f"{home_error:.2f} rad). The controllers are probably not "
                "active - check for a stale gzserver from a previous run."
            )
        self._check_gripper_ready()
        self._ready_pub.publish(Bool(data=True))
        self._ready_event.set()
        self.get_logger().info("execution ready")

    # ---------------- IK helpers ----------------
    def _tool_pose(self, pos, rot):
        return pose_from_rot_trans(rot, pos)

    def _solve(self, target_t, link=None):
        link = link or self.tool_link
        seeds = [self._current_q] + self.seed_sets
        return self.model.ik_retry(target_t, seeds, link=link)

    def _solve_home(self):
        pos = self.get_parameter("home_camera_pos").value
        look = self.get_parameter("home_camera_look").value
        cam_target = pose_from_rot_trans(look_at_matrix(pos, look), pos)
        tool_target = cam_target @ self.model.camera_in_tool()
        seeds = [self._current_q] + self.seed_sets
        q = self.model.ik_retry(tool_target, seeds)
        if q is None:
            raise RuntimeError("home pose IK failed; check home_camera_pos")
        return q

    # ---------------- low-level motion ----------------
    def _move_to_joints(self, target_q, duration):
        if duration <= 0:
            return True
        target_q = np.asarray(target_q, dtype=float)
        if not np.all(np.isfinite(target_q)):
            self.get_logger().warn("refusing to send a non-finite trajectory")
            return False
        target_q = nearest_joint_branch(
            target_q, self._current_q, self.joint_lower, self.joint_upper
        )
        # Stay strictly inside the joint limits: the controller rejects
        # trajectories that touch them, and a state message can report a joint
        # a hair beyond its limit.
        target_q = np.clip(target_q, self.joint_lower + 1e-6, self.joint_upper - 1e-6)
        waypoints = interpolate_joints(
            self._current_q, target_q, steps=self.traj_steps
        )
        waypoints = np.clip(
            waypoints, self.joint_lower + 1e-6, self.joint_upper - 1e-6
        )
        traj = JointTrajectory()
        traj.joint_names = list(self.model.arm_joints)
        for i, q in enumerate(waypoints):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.time_from_start = rclpy.duration.Duration(
                seconds=duration * i / (len(waypoints) - 1)
            ).to_msg()
            traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        for _ in range(6):
            tol = JointTolerance()
            # The default position loop in gazebo_ros2_control is weak
            # (gain 0.1): a 0.02 rad tolerance aborts trajectories that are
            # still converging. 0.05 rad is ~3 deg and well inside what the
            # grasp geometry tolerates.
            tol.position = 0.05
            tol.velocity = 0.10
            tol.name = self.model.arm_joints[_]
            goal.goal_tolerance.append(tol)
        # Give the position loop time to settle: in WSL a 0.5 s tolerance made
        # the controller abort trajectories that were still converging.
        goal.goal_time_tolerance = rclpy.duration.Duration(seconds=2.0).to_msg()

        future = self._arm_client.send_goal_async(goal)
        # A busy controller can take a while to answer, and giving up while it
        # still starts the trajectory would desynchronise our joint bookkeeping.
        if not self._wait_future(future, 30.0):
            self.get_logger().warn("arm controller did not answer the goal request")
            return False
        if future.result() is None:
            self.get_logger().warn("arm controller goal request returned nothing")
            return False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("arm controller rejected the trajectory")
            return False
        result_future = goal_handle.get_result_async()
        if not self._wait_future(result_future, duration + 30.0):
            self.get_logger().warn("timed out waiting for the arm controller result")
            return False
        if result_future.result() is None:
            self.get_logger().warn("arm controller returned an empty result")
            return False
        result = result_future.result().result
        if result.error_code not in (FollowJointTrajectory.Result.SUCCESSFUL, 0):
            # The controller sometimes reports a tolerance violation even
            # though the arm stopped exactly where we asked; check the joint
            # states before throwing the motion away.
            self._sleep(0.5)
            error = joint_error(self._current_q, target_q)
            self.get_logger().warn(
                f"arm controller returned error_code={int(result.error_code)}; "
                f"max joint error after settling is {error:.4f} rad"
            )
            if error > 0.08:
                return False
        self._current_q = np.array(target_q, dtype=float)
        return True

    def _set_gripper(self, positions, duration):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in positions]
        self._gripper_pub.publish(msg)
        self._sleep(duration)

    def _grasp_close(self, steps=8, duration=0.3):
        """Close the jaws gradually and stop as soon as they are blocked.

        A single hard close command shoves a light object out of the jaws
        (measured: the object moved ~5 cm). Closing in small increments and
        watching /joint_states lets us stop at first contact, which is what a
        real parallel gripper does with a force limit.
        """
        target = float(self.params["gripper_close"][0])
        # Always start from a fully open, settled gripper: the previous action
        # may still be opening, and a moving start made the width-based test
        # fire on the first step with a negative "movement".
        self._set_gripper(self.params["gripper_open"], 0.8)
        per_step = target / steps
        for i in range(1, steps + 1):
            q = per_step * i
            self._set_gripper([q, q], duration)
            positions = self._joint_positions
            left = positions.get("finger_left")
            right = positions.get("finger_right")
            if left is None or right is None:
                continue
            achieved = min(left, right)
            # Blocked = the joint could not follow the command. Comparing the
            # *achieved* position with the command is unambiguous, unlike the
            # width derivative (which is noisy while the jaws accelerate).
            if achieved < 0.6 * q:
                self.get_logger().info(
                    f"jaw contact at commanded q={q:.4f} m, achieved "
                    f"{achieved:.4f} m, width={self._gripper_width():.4f} m"
                )
                return True
        return self._is_holding()

    def _gripper_width(self):
        try:
            q_sum = (
                self._joint_positions["finger_left"]
                + self._joint_positions["finger_right"]
            )
        except KeyError:
            return 0.0
        return gripper_width(
            self.params,
            self._joint_positions["finger_left"],
            self._joint_positions["finger_right"],
        )

    def _is_holding(self):
        """An object is between the jaws when they cannot reach the closed gap."""
        try:
            return is_holding(
                self.params,
                self._joint_positions["finger_left"],
                self._joint_positions["finger_right"],
            )
        except KeyError:
            return False

    def _log_tool_pose(self, target_t):
        """Compare the commanded tool pose with the one reported by TF.

        This is what tells us whether a failed grasp is caused by the arm not
        reaching the target or by something upstream (height map, object
        placement). TF comes from robot_state_publisher + /joint_states, i.e.
        the joint positions the robot actually reports.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_link, self.tool_link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            self.get_logger().warn(
                f"no TF {self.base_link}->{self.tool_link}: {type(exc).__name__}"
            )
            return
        t = transform.transform.translation
        actual = np.array([t.x, t.y, t.z], dtype=float)
        target = np.asarray(target_t, dtype=float)[:3, 3]
        error = float(np.linalg.norm(actual - target))
        message = (
            f"tool pose: commanded ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}) "
            f"actual ({actual[0]:.3f}, {actual[1]:.3f}, {actual[2]:.3f}) "
            f"error={error * 1000.0:.1f} mm"
        )
        if error > 0.02:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def _log_objects_near(self, x, y, top=3):
        """Report where gazebo thinks the clutter really is (debug aid).

        The height map and the arm can both be right while the object has
        already been nudged away; only gazebo's own entity state can tell.
        """
        if not self.verify_objects:
            return
        if not self._world_props_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                "gazebo_ros_state services unavailable; cannot verify objects"
            )
            return
        request = GetWorldProperties.Request()
        future = self._world_props_cli.call_async(request)
        if not self._wait_future(future, 3.0) or future.result() is None:
            return
        names = [n for n in future.result().model_names if n.startswith("ag_obj_")]
        found = []
        for name in names:
            entity = GetEntityState.Request()
            entity.name = name
            entity.reference_frame = "world"
            entity_future = self._entity_state_cli.call_async(entity)
            if not self._wait_future(entity_future, 2.0):
                continue
            result = entity_future.result()
            if result is None or not result.success:
                continue
            position = result.state.position
            found.append(
                (float(np.hypot(position.x - x, position.y - y)), name, position)
            )
        found.sort(key=lambda item: item[0])
        for distance, name, position in found[:top]:
            self.get_logger().info(
                f"object {name}: world ({position.x:.3f}, {position.y:.3f}, "
                f"z={position.z:.3f}), distance_to_grasp={distance * 1000.0:.0f} mm"
            )

    def _step(self, step):
        kind = step["kind"]
        if kind == "move":
            target = self._tool_pose(step["pos"], step["rot"])
            q = self._solve(target)
            if q is None:
                self.get_logger().warn(f"IK found no solution for {step['pos']}")
                return False
            if not self._move_to_joints(q, step["dur"]):
                self.get_logger().warn(f"trajectory did not reach {step['pos']}")
                return False
            self._log_tool_pose(target)
            return True
        if kind == "gripper":
            if np.allclose(step["pos"], self.params["gripper_close"]):
                self._grasp_close()
            else:
                self._set_gripper(step["pos"], step["dur"])
            return True
        if kind == "hold_check":
            self._sleep(step["dur"])
            self._hold_width = self._gripper_width()
            return self._is_holding()
        if kind == "view":
            cam_target = pose_from_rot_trans(
                look_at_matrix(step["pos"], step["look"]), step["pos"]
            )
            tool_target = cam_target @ self.model.camera_in_tool()
            q = self._solve(tool_target)
            if q is None:
                self.get_logger().warn(f"IK found no solution for view {step['pos']}")
                return False
            if not self._move_to_joints(q, step["dur"]):
                return False
            self._sleep(step["dwell"])
            req = Observe.Request()
            req.auxiliary_view = True
            future = self._observe_cli.call_async(req)
            return self._wait_future(future, 15.0) and future.result() is not None
        if kind == "home":
            return self._move_to_joints(self.home_q, step["dur"])
        raise ValueError(f"unknown plan step kind: {kind}")

    # ---------------- services ----------------
    def _home_cb(self, request, response):
        if not self._wait_ready():
            response.success = False
            response.message = "not ready"
            return response
        try:
            response.success = self._move_to_joints(self.home_q, 1.2)
        except Exception as exc:
            self.get_logger().error(f"go_home crashed: {type(exc).__name__}: {exc}")
            response.success = False
        response.message = "home reached" if response.success else "home failed"
        return response

    def _grasp_cb(self, request, response):
        response.success = False
        response.object_cleared = False
        response.gripper_width = 0.0
        self._hold_width = 0.0
        if not self._wait_ready():
            return response
        try:
            u, v = int(request.u), int(request.v)
            z_top = float(request.height)
            if z_top < self.min_grasp_height:
                return response
            x, y = grid_to_xy(self.grid, u, v)
            plan = grasp_plan(self.params, x, y, z_top)

            held = False
            abort = False
            for step in plan:
                if step["kind"] == "hold_check":
                    held = self._step(step)
                    if not held:
                        abort = True
                        break
                else:
                    if not self._step(step):
                        abort = True
                        break

            if abort:
                self._set_gripper(self.params["gripper_open"], 0.3)
                if not self._move_to_joints(self.home_q, 1.2):
                    self.get_logger().warn("recovery: home move failed")

            response.success = held
            response.object_cleared = held
            response.gripper_width = self._gripper_width()
            self.get_logger().info(
                f"grasp cell ({u},{v}) -> ({x:.3f}, {y:.3f}), z_top={z_top:.3f}, "
                f"descend={plan[1]['pos'][2]:.3f}, held={held}, "
                f"width_at_hold={self._hold_width:.4f} m "
                f"(closed={self.params['gripper_closed_gap']:.3f}, "
                f"open={self.params['gripper_open_gap']:.3f})"
            )
            if not held:
                self._log_objects_near(x, y)
        except Exception as exc:
            self.get_logger().error(
                f"grasp crashed: {type(exc).__name__}: {exc}; recovering"
            )
            try:
                self._move_to_joints(self.home_q, 1.2)
            except Exception:
                pass
        return response

    def _push_cb(self, request, response):
        response.success = False
        if not self._wait_ready():
            return response
        try:
            u, v = int(request.u), int(request.v)
            direction = int(request.direction) % 8
            z_top = float(request.height)
            if z_top < self.min_grasp_height:
                return response
            x, y = grid_to_xy(self.grid, u, v)
            theta = direction * math.pi / 4.0
            plan = push_plan(self.params, x, y, theta)

            ok = True
            for step in plan:
                if not self._step(step):
                    ok = False
                    break
            if not ok:
                if not self._move_to_joints(self.home_q, 1.2):
                    self.get_logger().warn("recovery: home move failed")
            response.success = ok
        except Exception as exc:
            self.get_logger().error(f"push crashed: {type(exc).__name__}: {exc}")
        return response

    def _view_cb(self, request, response):
        response.success = False
        if not self._wait_ready():
            return response
        try:
            view_id = int(request.view_id)
            views = self.params["views"]
            if not 0 <= view_id < len(views):
                return response
            plan = view_plan(self.params, view_id)
            ok = True
            for step in plan:
                if not self._step(step):
                    ok = False
                    break
            if not ok:
                if not self._move_to_joints(self.home_q, 1.2):
                    self.get_logger().warn("recovery: home move failed")
            response.success = ok
        except Exception as exc:
            self.get_logger().error(f"move_view crashed: {type(exc).__name__}: {exc}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ExecutionNode()
    # A multi threaded executor keeps delivering action results, joint states
    # and service responses while a grasp/push callback is blocking. Without
    # it the node had to spin itself from inside its own callback, which
    # silently stopped working and left every goal unanswered.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    node._external_spin = True

    def _spin():
        try:
            executor.spin()
        except Exception:
            pass

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()
    try:
        node.startup()
        spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
