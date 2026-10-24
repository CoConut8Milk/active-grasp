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
    """Fingers pointing down, closing axis along world x.

    The fingers hang along **-tool z** (their joint sits at z=-0.01 and the
    pads extend to z=-0.07 in the tool frame). Pointing tool z *down* therefore
    flips the gripper upside down: the pads sit 1-7 cm ABOVE tool0 and can
    never touch an object on the table. Keep tool z up (identity) so the pads
    hang below the flange.
    """
    return np.eye(3)


def gripper_width(params, q_left, q_right):
    """Distance between the finger pads for the two prismatic joint values."""
    return params["gripper_open_gap"] - (q_left + q_right)


def is_holding(params, q_left, q_right):
    """True when the closed jaws are blocked by something between them.

    A fully closed gripper reaches ``gripper_closed_gap``; a fully open one
    reaches ``gripper_open_gap``. Anything strictly between the two means the
    fingers stopped early, i.e. an object is being squeezed.
    """
    width = gripper_width(params, q_left, q_right)
    return (
        params["gripper_closed_gap"] + params["held_margin"]
        < width
        < params["gripper_open_gap"] - 0.005
    )


def push_orientation(theta):
    """Fingers pointing down; pads straddle the object across the push line."""
    s, c = math.sin(theta), math.cos(theta)
    # x and y span the horizontal plane, z stays up (see grasp_orientation).
    return np.column_stack([[s, -c, 0.0], [c, s, 0.0], [0.0, 0.0, 1.0]])


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
    # The fingers reach finger_length below tool0. Descend until the tips are
    # ~6 mm above the table: that straddles every object in the scene (2-4.5 cm
    # tall) and, unlike a target derived from z_top, does not depend on the
    # accuracy of the height map. A constant tool height used to pinch only the
    # top edge, which always slips.
    finger_length = params.get("finger_length", 0.07)
    descend_z = 0.006 + finger_length
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

