import math

import numpy as np

from ag_execution.primitives import (
    grasp_orientation,
    grasp_plan,
    grid_to_xy,
    gripper_width,
    interpolate_joints,
    is_holding,
    push_orientation,
    push_plan,
    view_plan,
)


def _params():
    return {
        "bin_pos": [0.45, -0.34, 0.32],
        "views": [
            {"pos": [0.45, 0.0, 0.72], "look": [0.45, 0.0, 0.02]}
        ],
        "gripper_close": [0.034, 0.034],
        "gripper_open": [0.0, 0.0],
        "gripper_open_gap": 0.070,
        "gripper_closed_gap": 0.002,
        "held_margin": 0.004,
        "approach_offset": 0.18,
        "grasp_offset": 0.065,
        "finger_length": 0.08,
        "lift_height": 0.30,
        "push_height": 0.075,
        "push_length": 0.09,
        "view_dwell": 1.2,
        "dur_approach": 0.9, "dur_descend": 0.55, "dur_close": 0.7,
        "dur_hold_check": 0.35, "dur_lift": 0.6, "dur_transit": 0.9,
        "dur_open": 0.5, "dur_home": 1.0, "dur_push": 0.7,
        "dur_retreat": 0.5, "dur_view": 1.0,
    }


def test_grid_to_xy_center_and_corners():
    grid = {"x_min": 0.2, "x_max": 0.7, "y_min": -0.25, "y_max": 0.25,
            "width": 48, "height": 48}
    x, y = grid_to_xy(grid, 0, 0)
    assert abs(x - 0.205) < 1e-3 and abs(y - 0.245) < 1e-3
    x, y = grid_to_xy(grid, 47, 47)
    assert abs(x - 0.695) < 1e-3 and abs(y + 0.245) < 1e-3


def test_push_orientation_is_proper_rotation():
    import math
    r = push_orientation(math.pi / 4)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(r) > 0
    # tool z stays up so the fingers (along -z) hang down towards the table
    assert np.allclose(r[:, 2], [0, 0, 1])


def test_grasp_orientation_keeps_the_fingers_below_the_flange():
    r = grasp_orientation()
    assert np.linalg.det(r) > 0
    # +tool z -> the fingers are at -tool z, so this must point up
    assert r[2, 2] > 0.9


def test_plans_are_well_formed():
    params = _params()
    kinds_grasp = [s["kind"] for s in grasp_plan(params, 0.45, 0.0, 0.05)]
    assert kinds_grasp[0] == "move" and "hold_check" in kinds_grasp
    assert grasp_plan(params, 0.45, 0.0, 0.05)[-1]["kind"] == "home"

    push = push_plan(params, 0.45, 0.0, math.pi / 2)
    assert push[1]["pos"][1] > push[0]["pos"][1]  # pushed toward +y
    assert push[-1]["kind"] == "home"

    view = view_plan(params, 0)
    assert view[0]["kind"] == "view" and view[-1]["kind"] == "home"


def test_grasp_descends_so_fingers_straddle_the_object():
    params = _params()
    finger_length = params["finger_length"]
    for z_top in (0.022, 0.035, 0.045):
        plan = grasp_plan(params, 0.45, 0.0, z_top)
        descend_z = plan[1]["pos"][2]
        tip_z = descend_z - finger_length
        assert tip_z > 0.005, "fingertips must stay above the table"
        assert tip_z < z_top - 0.005, "fingertips must go below the object top"
        approach_z = plan[0]["pos"][2]
        assert approach_z > descend_z


def test_hold_detection_uses_the_real_finger_geometry():
    params = _params()
    # Wide open: nothing between the jaws yet.
    assert abs(gripper_width(params, 0.0, 0.0) - 0.070) < 1e-9
    assert not is_holding(params, 0.0, 0.0)
    # Fully closed on air: still not holding.
    assert abs(gripper_width(params, 0.034, 0.034) - 0.002) < 1e-9
    assert not is_holding(params, 0.034, 0.034)
    # A 3 cm object stops each finger at 2.0 cm -> 3 cm remains.
    assert abs(gripper_width(params, 0.020, 0.020) - 0.030) < 1e-9
    assert is_holding(params, 0.020, 0.020)


def test_interpolate_joints_endpoints():
    q0 = np.zeros(6)
    q1 = np.ones(6)
    waypoints = interpolate_joints(q0, q1, steps=5)
    assert len(waypoints) == 5
    assert np.allclose(waypoints[0], q0)
    assert np.allclose(waypoints[-1], q1)
