"""Procedural clutter manager.

Exposes the ag_interfaces/Reset service. Each reset deletes every object
spawned so far and rebuilds the scene from random primitives, which gives
the agent a fresh episode with domain randomization for free.
"""

import math
import random
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose
from ag_interfaces.srv import Reset


def _box_inertia(mass, sx, sy, sz):
    return (
        mass / 12.0 * (sy * sy + sz * sz),
        mass / 12.0 * (sx * sx + sz * sz),
        mass / 12.0 * (sx * sx + sy * sy),
    )


def _cylinder_inertia(mass, radius, height):
    return (
        mass / 12.0 * (3.0 * radius * radius + height * height),
        mass / 12.0 * (3.0 * radius * radius + height * height),
        mass * radius * radius / 2.0,
    )


def build_object_sdf(name, rng):
    """Return an SDF string for one small clutter object."""
    shape = rng.choice(["box", "cylinder", "sphere"])
    color = [rng.random(), rng.random(), rng.random()]
    mu = rng.uniform(0.45, 1.0)
    density = rng.uniform(500.0, 1200.0)

    geometry = ""
    mass = 0.0
    ixx = iyy = izz = 0.0

    if shape == "box":
        # Keep every object narrower than the 4 cm open gripper gap, otherwise
        # the fingers collide with the object instead of closing around it.
        sx = rng.uniform(0.020, 0.032)
        sy = rng.uniform(0.020, 0.034)
        sz = rng.uniform(0.020, 0.040)
        mass = density * sx * sy * sz
        ixx, iyy, izz = _box_inertia(mass, sx, sy, sz)
        geometry = f'<box><size>{sx} {sy} {sz}</size></box>'
    elif shape == "cylinder":
        radius = rng.uniform(0.010, 0.016)
        height = rng.uniform(0.025, 0.045)
        mass = density * math.pi * radius * radius * height
        ixx, iyy, izz = _cylinder_inertia(mass, radius, height)
        geometry = f'<cylinder><radius>{radius}</radius><length>{height}</length></cylinder>'
    else:
        radius = rng.uniform(0.012, 0.017)
        mass = density * 4.0 / 3.0 * math.pi * radius**3
        ixx = iyy = izz = 0.4 * mass * radius * radius
        geometry = f'<sphere><radius>{radius}</radius></sphere>'

    return f"""<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <link name="link">
      <inertial>
        <mass>{mass:.5f}</mass>
        <inertia>
          <ixx>{ixx:.7f}</ixx>
          <iyy>{iyy:.7f}</iyy>
          <izz>{izz:.7f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>{geometry}</geometry>
        <surface>
          <friction>
            <ode>
              <mu>{mu:.3f}</mu>
              <mu2>{mu:.3f}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>{geometry}</geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} 1</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class WorldManager(Node):
    def __init__(self):
        super().__init__("world_manager")

        self.declare_parameter("workspace_x_min", 0.20)
        self.declare_parameter("workspace_x_max", 0.70)
        self.declare_parameter("workspace_y_min", -0.25)
        self.declare_parameter("workspace_y_max", 0.25)
        self.declare_parameter("table_top_z", 0.78)
        # Keep clutter inside the region the arm can actually reach with the
        # tightened joint limits. Verified by exhaustive IK over this box.
        self.declare_parameter("spawn_x_min", 0.40)
        self.declare_parameter("spawn_x_max", 0.62)
        self.declare_parameter("spawn_y_min", -0.12)
        self.declare_parameter("spawn_y_max", 0.12)
        # Kept for backwards compatibility with older parameter files.
        self.declare_parameter("spawn_margin", 0.06)
        self.declare_parameter("max_spawn_attempts", 40)
        self.declare_parameter("settle_time", 1.2)
        self.declare_parameter("seed", 0)

        self.workspace = {
            "x_min": self.get_parameter("workspace_x_min").value,
            "x_max": self.get_parameter("workspace_x_max").value,
            "y_min": self.get_parameter("workspace_y_min").value,
            "y_max": self.get_parameter("workspace_y_max").value,
        }
        self.table_top_z = self.get_parameter("table_top_z").value
        self.spawn_margin = self.get_parameter("spawn_margin").value
        self.spawn = {
            "x_min": self.get_parameter("spawn_x_min").value,
            "x_max": self.get_parameter("spawn_x_max").value,
            "y_min": self.get_parameter("spawn_y_min").value,
            "y_max": self.get_parameter("spawn_y_max").value,
        }
        self.max_attempts = self.get_parameter("max_spawn_attempts").value
        self.settle_time = self.get_parameter("settle_time").value
        self.seed = self.get_parameter("seed").value
        self._rng = random.Random(self.seed)

        # Blocking gazebo service calls are made from a dedicated helper node:
        # the main node is single threaded and busy inside the reset callback,
        # so spinning it re-entrantly would be unsafe.
        self._io_node = rclpy.create_node("world_manager_io")
        self._spawn_cli = self._io_node.create_client(SpawnEntity, "/spawn_entity")
        self._delete_cli = self._io_node.create_client(DeleteEntity, "/delete_entity")
        self._reset_srv = self.create_service(Reset, "~/reset_scene", self._on_reset)

        self._spawned_names = []
        self._object_counter = 0

        for cli in (self._spawn_cli, self._delete_cli):
            if not cli.wait_for_service(timeout_sec=20.0):
                self.get_logger().error(
                    f"service {cli.srv_name} unavailable; is gazebo running?"
                )

        self.get_logger().info("world manager ready")

    def _call_blocking(self, cli, request, timeout=5.0):
        future = cli.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self._io_node, timeout_sec=0.02)
        if not future.done() or future.result() is None:
            return None
        return future.result()

    def _delete_all(self):
        for name in list(self._spawned_names):
            req = DeleteEntity.Request()
            req.name = name
            self._call_blocking(self._delete_cli, req)
        self._spawned_names.clear()

    def _spawn_one(self, name, x, y, z, yaw):
        req = SpawnEntity.Request()
        req.name = name
        req.xml = build_object_sdf(name, self._rng)
        req.initial_pose = Pose()
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.initial_pose.orientation.z = math.sin(yaw / 2.0)
        req.initial_pose.orientation.w = math.cos(yaw / 2.0)
        return self._call_blocking(self._spawn_cli, req) is not None

    def _sample_poses(self, n):
        """Rejection-sample n non-overlapping poses inside the workspace."""
        poses = []
        w = self.spawn
        margin = self.spawn_margin
        for _ in range(n):
            for _ in range(self.max_attempts):
                x = self._rng.uniform(
                    w["x_min"] + margin, w["x_max"] - margin
                )
                y = self._rng.uniform(
                    w["y_min"] + margin, w["y_max"] - margin
                )
                if all((x - px) ** 2 + (y - py) ** 2 >= 0.012 for px, py in poses):
                    poses.append((x, y))
                    break
            else:
                poses.append((x, y))
        return poses

    def _on_reset(self, request, response):
        # Gazebo may still be starting up when the first reset arrives; the
        # factory service only appears once gzserver loaded its plugins.
        if not self._spawn_cli.service_is_ready():
            self.get_logger().warn("waiting for /spawn_entity to appear ...")
            self._spawn_cli.wait_for_service(timeout_sec=90.0)
            self.get_logger().info("/spawn_entity is available")
        self._delete_all()
        n = max(1, min(int(request.num_objects), 12))

        for x, y in self._sample_poses(n):
            yaw = self._rng.uniform(-math.pi, math.pi)
            name = f"ag_obj_{self._object_counter}"
            self._object_counter += 1
            ok = self._spawn_one(name, x, y, self.table_top_z + 0.03, yaw)
            if ok:
                self._spawned_names.append(name)
                self.get_logger().info(
                    f"spawned {name} at ({x:.2f}, {y:.2f})"
                )
            else:
                self.get_logger().warn(f"could not spawn {name}")

        # Let objects fall and settle before the agent observes the scene.
        # Wall-clock sleep: in realtime mode this equals sim time, and in the
        # fast world physics settles even sooner than real time.
        time.sleep(self.settle_time)

        response.success = True
        response.objects_spawned = len(self._spawned_names)
        self.get_logger().info(
            f"scene ready: {response.objects_spawned} objects on the table"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WorldManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._io_node.destroy_node()
        except AttributeError:
            pass
        node.destroy_node()
        rclpy.shutdown()
