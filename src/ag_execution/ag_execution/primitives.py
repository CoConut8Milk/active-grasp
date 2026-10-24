"""Motion primitive plans (pure geometry, no ROS).

Each plan is a list of steps consumed by execution_node:
  {"kind": "move", "pos": [...], "rot": 3x3, "dur": seconds}
  {"kind": "gripper", "pos": [l, r], "dur": seconds}
  {"kind": "hold_check", "dur": seconds}
  {"kind": "view", "pos": [...], "look": [...], "dur": ..., "dwell": ...}
  {"kind": "home", "dur": seconds}
"""

import math

import numpy as np


def grasp_orientation():
    """Tool z straight down, tool x forward: Rx(pi)."""
    return np.diag([1.0, -1.0, -1.0])


def push_orientation(theta):
    """Tool z down; finger axis perpendicular to the push direction."""
    s, c = math.sin(theta), math.cos(theta)
    return np.column_stack([[-s, c, 0.0], [c, s, 0.0], [0.0, 0.0, -1.0]])


def grid_to_xy(grid, u, v):
    x_min, x_max = grid["x_min"], grid["x_max"]
    y_min, y_max = grid["y_min"], grid["y_max"]
    res = (x_max - x_min) / grid["width"]
    x = x_min + (u + 0.5) * res
    y = y_max - (v + 0.5) * res
    return x, y


def grasp_plan(params, x, y, z_top):
    orient = grasp_orientation()
    bin_pos = params["bin_pos"]
    # The fingers reach finger_length below tool0. Aim the fingertips at half
    # the object height (and never below 1 cm above the table) so that the
    # object is actually between the jaws instead of being pinched by its top
    # edge - with a constant tool height the fingertips only dip 5 mm below
    # the object top for every object size.
    finger_length = params.get("finger_length", 0.07)
    tip_z = max(0.01, 0.5 * z_top)
    descend_z = tip_z + finger_length
    return [
        {"kind": "move", "pos": [x, y, descend_z + params["approach_offset"]],
         "rot": orient, "dur": params["dur_approach"]},
        {"kind": "move", "pos": [x, y, descend_z],
         "rot": orient, "dur": params["dur_descend"]},
        {"kind": "gripper", "pos": params["gripper_close"], "dur": params["dur_close"]},
        {"kind": "hold_check", "dur": params["dur_hold_check"]},
        {"kind": "move", "pos": [x, y, params["lift_height"]],
         "rot": orient, "dur": params["dur_lift"]},
        {"kind": "move", "pos": bin_pos, "rot": orient, "dur": params["dur_transit"]},
        {"kind": "gripper", "pos": params["gripper_open"], "dur": params["dur_open"]},
        {"kind": "home", "dur": params["dur_home"]},
    ]


def push_plan(params, x, y, theta):
    start = [x, y, params["push_height"]]
    end = [
        x + params["push_length"] * math.cos(theta),
        y + params["push_length"] * math.sin(theta),
        params["push_height"],
    ]
    retreat = [x, y, params["push_height"] + 0.10]
    rot = push_orientation(theta)
    return [
        {"kind": "move", "pos": start, "rot": rot, "dur": params["dur_approach"]},
        {"kind": "move", "pos": end, "rot": rot, "dur": params["dur_push"]},
        {"kind": "move", "pos": retreat, "rot": rot, "dur": params["dur_retreat"]},
        {"kind": "home", "dur": params["dur_home"]},
    ]


def view_plan(params, view_id):
    view = params["views"][view_id]
    return [
        {"kind": "view", "pos": view["pos"], "look": view["look"],
         "dur": params["dur_view"], "dwell": params["view_dwell"]},
        {"kind": "home", "dur": params["dur_home"]},
    ]


def interpolate_joints(q0, q1, steps=8):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    return [q0 + (q1 - q0) * t for t in np.linspace(0.0, 1.0, steps)]

