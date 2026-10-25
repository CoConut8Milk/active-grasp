"""ROS node: fuses the wrist RGB-D stream into height + uncertainty maps.

Services:
  ~/observe      (ag_interfaces/Observe)  capture + fuse + return state
  ~/clear        (std_srvs/Trigger)       clear the whole world model
  ~/clear_aux    (std_srvs/Trigger)       drop the auxiliary-view snapshot

At startup the node waits for the empty table, auto-calibrates the camera's
image mirroring, and only then accepts requests.
"""

import math
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from ag_interfaces.srv import Observe
from ag_perception.fusion import (
    HeightmapFusion, auto_calibrate, backproject, pose_to_matrix,
)


def _stamp_key(stamp):
    return (stamp.sec, stamp.nanosec)


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("depth_topic", "/ag_camera/depth/image_raw")
        self.declare_parameter("color_topic", "/ag_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/ag_camera/depth/camera_info")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("grid_height", 48)
        self.declare_parameter("grid_width", 48)
        self.declare_parameter("workspace_x_min", 0.20)
        self.declare_parameter("workspace_x_max", 0.70)
        self.declare_parameter("workspace_y_min", -0.25)
        self.declare_parameter("workspace_y_max", 0.25)
        self.declare_parameter("z_max", 0.35)
        self.declare_parameter("max_age", 8)
        self.declare_parameter("stationary_velocity", 0.08)
        self.declare_parameter("fresh_frame_timeout", 1.5)
        self.declare_parameter("calib_max_depth", 1.1)
        # Pixels this close to the image border carry unreliable depth in
        # Gazebo Classic and would otherwise show up as permanent tall cells.
        self.declare_parameter("valid_border", 3)
        # Image axis convention of the simulator (see perception.yaml).
        self.declare_parameter("u_sign", -1.0)
        self.declare_parameter("v_sign", 1.0)
        self.declare_parameter("hfov", 1.047197551)
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)

        p = self.get_parameter
        self.depth_topic = p("depth_topic").value
        self.color_topic = p("color_topic").value
        self.camera_info_topic = p("camera_info_topic").value
        self.joint_states_topic = p("joint_states_topic").value
        self.base_frame = p("base_frame").value
        self.camera_frame = p("camera_frame").value
        self.stationary_velocity = p("stationary_velocity").value
        self.fresh_timeout = p("fresh_frame_timeout").value
        self.calib_max_depth = p("calib_max_depth").value
        self.valid_border = int(p("valid_border").value)
        self.u_sign = float(p("u_sign").value)
        self.v_sign = float(p("v_sign").value)
        self.hfov = p("hfov").value
        self.img_w = p("image_width").value
        self.img_h = p("image_height").value

        self.fusion = HeightmapFusion(
            p("grid_height").value, p("grid_width").value,
            p("workspace_x_min").value, p("workspace_x_max").value,
            p("workspace_y_min").value, p("workspace_y_max").value,
            z_max=p("z_max").value, max_age=p("max_age").value,
        )

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._cond = threading.Condition()
        self._latest_depth = None
        self._latest_color = None
        self._color_size = None
        self._latest_joint_vel = {}
        self._joint_pos = {}
        self._joint_time = None
        self._joint_speed = 0.0
        self._joint_speed_joint = ""
        self._joint_names_logged = False
        self._last_used_stamp = None
        self._k_matrix = None
        self._signs = None
        self._execution_ready = False
        self._ready = False
        self._ready_warned = False
        # Startup diagnostics: how long we tolerate the missing
        # /ag_execution/ready latch, and how often we explain what we wait for.
        self.ready_grace = 20.0
        self.status_period = 3.0
        self._start_time = time.monotonic()
        self._last_status = 0.0
        self._still_since = None

        qos = qos_profile_sensor_data
        # Sensor callbacks live in a reentrant group so depth frames keep
        # arriving while the (blocking) observe service waits for a fresh one.
        self._sensor_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image, self.depth_topic, self._depth_cb, qos,
            callback_group=self._sensor_group,
        )
        self.create_subscription(
            Image, self.color_topic, self._color_cb, qos,
            callback_group=self._sensor_group,
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._info_cb, qos,
            callback_group=self._sensor_group,
        )
        self.create_subscription(
            JointState, self.joint_states_topic, self._joint_cb, qos,
            callback_group=self._sensor_group,
        )
        ready_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Bool, "/ag_execution/ready", self._ready_cb, ready_qos,
            callback_group=self._sensor_group,
        )

        self._observe_srv = self.create_service(Observe, "~/observe", self._observe_cb)
        self._clear_srv = self.create_service(Trigger, "~/clear", self._clear_cb)
        self._clear_aux_srv = self.create_service(Trigger, "~/clear_aux", self._clear_aux_cb)

        # Startup calibration runs from a timer so the constructor never
        # blocks the executor; the robot must be at home on the empty table.
        self._startup_timer = self.create_timer(0.5, self._startup_tick)
        self.get_logger().info(
            f"perception node up; depth={self.depth_topic} "
            f"color={self.color_topic} joints={self.joint_states_topic}"
        )
        self.get_logger().info("waiting for depth frames and a settled robot...")

    # ---------------- subscriptions ----------------
    def _depth_cb(self, msg):
        with self._cond:
            self._latest_depth = msg
            self._cond.notify_all()

    def _color_cb(self, msg):
        with self._cond:
            self._latest_color = msg
            self._color_size = (msg.height, msg.width)
            self._cond.notify_all()

    def _info_cb(self, msg):
        k = np.array(msg.k).reshape(3, 3)
        if k[0, 0] > 0:
            self._k_matrix = k

    def _joint_cb(self, msg):
        now = time.monotonic()
        positions = dict(zip(msg.name, msg.position))
        with self._cond:
            if self._joint_pos and self._joint_time is not None:
                dt = max(now - self._joint_time, 1e-3)
                speeds = {
                    name: abs(positions[name] - self._joint_pos[name]) / dt
                    for name in positions if name in self._joint_pos
                }
                if speeds:
                    self._joint_speed_joint = max(speeds, key=speeds.get)
                    self._joint_speed = speeds[self._joint_speed_joint]
            self._joint_pos = positions
            self._joint_time = now
            self._latest_joint_vel = dict(zip(msg.name, msg.velocity))
            first_names = (
                sorted(positions)
                if positions and not self._joint_names_logged
                else None
            )
            if first_names is not None:
                self._joint_names_logged = True
        if first_names is not None:
            self.get_logger().info(
                f"joint states seen ({len(first_names)}): {first_names}"
            )

    def _ready_cb(self, msg):
        self._execution_ready = bool(msg.data)

    def _status(self, text):
        now = time.monotonic()
        if now - self._last_status < self.status_period:
            return
        self._last_status = now
        self.get_logger().warn(f"not calibrated yet: {text}")

    def _motion(self):
        """Largest joint speed we can measure, plus a human readable source.

        Gazebo Classic differentiates joint positions to report velocities,
        which leaves a noisy offset behind when the simulation runs below real
        time. The position delta measured against the wall clock is therefore
        preferred, with the published velocity array as a fallback so that
        readiness never depends on a single message field.
        """
        with self._cond:
            velocities = [
                (abs(float(v)), name) for name, v in self._latest_joint_vel.items()
                if math.isfinite(float(v))
            ]
            speed = self._joint_speed
            speed_joint = self._joint_speed_joint
        if self._joint_pos:
            return speed, f"joint '{speed_joint}' position delta"
        if velocities:
            value, name = max(velocities)
            return value, f"joint '{name}' velocity"
        return None, "no /joint_states yet"

    def _startup_tick(self):
        if self._ready:
            return
        with self._cond:
            has_depth = self._latest_depth is not None
        if not has_depth:
            self._status(f"waiting for depth frames on {self.depth_topic}")
            return
        speed, how = self._motion()
        if speed is None:
            self._status(how)
            return
        if speed > self.stationary_velocity:
            self._still_since = None
            self._status(f"robot still moving ({how} = {speed:.3f})")
            return
        now = time.monotonic()
        if self._still_since is None:
            self._still_since = now
            return
        if now - self._still_since < 1.5:
            return
        if not self._execution_ready and now - self._start_time < self.ready_grace:
            self._status("waiting for /ag_execution/ready (arm still going home)")
            return
        try:
            self._calibrate()
        except RuntimeError as exc:
            # Most often the camera is not facing the table yet: keep trying.
            self._still_since = None
            self._status(str(exc))
            return
        self._ready = True
        self._startup_timer.cancel()
        self.get_logger().info(
            f"perception ready (camera calibrated, signs={self._signs})"
        )

    # ---------------- helpers ----------------
    def _wait_stationary(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._latest_joint_vel:
                time.sleep(0.05)
                continue
            if max(abs(v) for v in self._latest_joint_vel.values()) < self.stationary_velocity:
                return
            time.sleep(0.05)
        self.get_logger().warn("robot did not settle; fusing anyway")

    def _wait_fresh_frame(self):
        with self._cond:
            deadline = time.monotonic() + self.fresh_timeout
            while True:
                msg = self._latest_depth
                if msg is not None:
                    key = _stamp_key(msg.header.stamp)
                    if self._last_used_stamp is None or key > self._last_used_stamp:
                        break
                if time.monotonic() >= deadline:
                    if msg is None:
                        return None
                    self.get_logger().warn("no fresh depth frame; reusing last one")
                    return msg
                self._cond.wait(0.05)
            self._last_used_stamp = _stamp_key(msg.header.stamp)
            return msg

    def _camera_pose(self, stamp):
        transform = None
        for target_time in (stamp, rclpy.time.Time()):
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.base_frame, self.camera_frame, target_time, timeout=rclpy.duration.Duration(seconds=0.5)
                )
                break
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        if transform is None:
            raise RuntimeError(
                f"TF {self.camera_frame} -> {self.base_frame} unavailable"
            )
        q = transform.transform.rotation
        t = transform.transform.translation
        return pose_to_matrix(
            [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
        )

    def _fallback_k(self):
        focal = self.img_w / (2.0 * np.tan(self.hfov / 2.0))
        return np.array(
            [[focal, 0, self.img_w / 2.0], [0, focal, self.img_h / 2.0], [0, 0, 1]]
        )

    def _calibrate(self):
        depth = self._latest_depth
        if depth is None:
            raise RuntimeError("no frame available for calibration")
        k = self._k_matrix if self._k_matrix is not None else self._fallback_k()
        pose = self._camera_pose(depth.header.stamp)
        depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="32FC1")
        est_u, est_v = auto_calibrate(
            depth_img, k, pose, max_depth=self.calib_max_depth
        )
        # A level table looks level in the mirrored cloud too, so the flatness
        # test can never observe the horizontal sign (measured: both u choices
        # give exactly zero height variance). Only the vertical sign carries
        # information - it tells us whether the camera is upside down. Take
        # that from the data, and the horizontal sign from it as well, since
        # u and v flip together when the camera rolls.
        self._signs = (self.u_sign, self.v_sign)
        if est_v != self.v_sign:
            self.get_logger().warn(
                f"flat-plane estimate says v_sign={est_v:+.0f} but the "
                f"configured value is {self.v_sign:+.0f}; adopting the "
                "estimate for both axes"
            )
            self._signs = (float(est_v), float(est_v))
        self.get_logger().info(
            f"camera signs: {self._signs} "
            f"(flat-plane estimate {(est_u, est_v)}, u unobservable)"
        )

    def _convert_color(self, color_msg, depth_shape):
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        if color.shape[:2] != depth_shape:
            out = np.zeros((*depth_shape, 3), dtype=np.uint8)
            h = min(color.shape[0], depth_shape[0])
            w = min(color.shape[1], depth_shape[1])
            out[:h, :w] = color[:h, :w]
            color = out
        return color

    # ---------------- services ----------------
    def _clear_cb(self, request, response):
        self.fusion.reset()
        response.success = True
        response.message = "world model cleared"
        return response

    def _clear_aux_cb(self, request, response):
        self.fusion.clear_aux()
        response.success = True
        response.message = "auxiliary view dropped"
        return response

    def _observe_cb(self, request, response):
        if not self._ready:
            if not self._ready_warned:
                self.get_logger().warn(
                    "observe called before calibration finished; returning empty state"
                )
                self._ready_warned = True
            response.height = Image()
            response.uncertainty = Image()
            response.color = Image()
            response.mask = Image()
            return response

        depth_msg = self._wait_fresh_frame()
        if depth_msg is None:
            response.height = Image()
            response.uncertainty = Image()
            response.color = Image()
            response.mask = Image()
            return response

        self._wait_stationary(timeout=2.0)
        k = self._k_matrix if self._k_matrix is not None else self._fallback_k()
        pose = self._camera_pose(depth_msg.header.stamp)
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        su, sv = self._signs

        points, valid = backproject(depth, k, pose, u_sign=su, v_sign=sv)
        b = self.valid_border
        if b > 0:
            valid[:b, :] = False
            valid[-b:, :] = False
            valid[:, :b] = False
            valid[:, -b:] = False
        color = self._latest_color
        if color is not None:
            color = self._convert_color(color, depth.shape)
        else:
            color = np.zeros((*depth.shape, 3), dtype=np.uint8)
        snapshot = self.fusion.build_snapshot(points, valid, color)
        height, uncertainty, color_map, mask = self.fusion.observe(
            snapshot, bool(request.auxiliary_view)
        )
        if not request.auxiliary_view:
            self._log_peaks(height)

        header = depth_msg.header
        response.height = self.bridge.cv2_to_imgmsg(height, encoding="32FC1")
        response.uncertainty = self.bridge.cv2_to_imgmsg(uncertainty, encoding="32FC1")
        response.color = self.bridge.cv2_to_imgmsg(color_map, encoding="rgb8")
        response.mask = self.bridge.cv2_to_imgmsg((mask * 255).astype(np.uint8), encoding="mono8")
        response.height.header = header
        response.uncertainty.header = header
        response.color.header = header
        response.mask.header = header
        return response

    def _log_peaks(self, height, top=3):
        """Report the tallest cells in world coordinates.

        Comparing these with the world manager's 'spawned ... at (x, y)'
        messages is the quickest way to tell whether the height map is
        geometrically correct or shifted/mirrored.
        """
        grid = np.asarray(height, dtype=np.float32).reshape(-1)
        if not np.any(grid > 0.005):
            return
        res = (self.fusion.x_max - self.fusion.x_min) / self.fusion.width
        parts = []
        for i in np.argsort(grid)[-top:][::-1]:
            if grid[i] <= 0.005:
                continue
            cj, ci = divmod(int(i), self.fusion.width)
            x = self.fusion.x_min + (ci + 0.5) * res
            y = self.fusion.y_max - (cj + 0.5) * res
            parts.append(f"({x:.3f}, {y:.3f}, z={grid[i]:.3f})")
        if parts:
            self.get_logger().info("heightmap peaks: " + ", ".join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
