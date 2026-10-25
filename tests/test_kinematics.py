import pathlib

import numpy as np

from ag_execution.kinematics import (
    RobotModel,
    joint_error,
    look_at_matrix,
    nearest_joint_branch,
    pose_from_rot_trans,
)

URDF = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "ag_description"
    / "urdf"
    / "ag_arm.urdf"
).read_text()


def _model():
    return RobotModel(URDF)


def _pose(rotation, position):
    return pose_from_rot_trans(rotation, position)


def test_fk_chain_geometry():
    model = _model()
    t = model.fk(np.zeros(6))
    # Vertical chain: tool0 is the sum of all z-offsets above base.
    assert np.allclose(t["tool0"][:3, 3], [0.0, 0.0, 1.22], atol=1e-3)
    # Camera sits to the +x side, viewing along +z (chain direction).
    cam = t["camera_link"]
    assert cam[:3, 3][0] > 0.05
    assert np.allclose(cam[:3, :3] @ np.array([1.0, 0, 0]), [0, 0, 1], atol=1e-4)


def test_ik_grasp_points_across_workspace():
    model = _model()
    rng = np.random.default_rng(0)
    orient = np.diag([1.0, -1.0, -1.0])
    seeds = [np.zeros(6), [0, -0.6, 1.4, -0.8, 0, 0]]

    for x in np.linspace(0.24, 0.66, 6):
        for y in np.linspace(-0.20, 0.20, 5):
            target = _pose(orient, [x, y, 0.095])
            q = model.ik_retry(target, seeds, random_seed=int(x * 100 + y * 100))
            assert q is not None, f"IK failed at ({x:.2f}, {y:.2f})"
            t = model.fk(q)["tool0"]
            assert np.linalg.norm(t[:3, 3] - [x, y, 0.095]) < 0.02


def test_fingertips_hang_below_the_tool_at_the_grasp_pose():
    """Regression: the old grasp orientation (tool z down) put the pads above."""
    from ag_execution.primitives import grasp_orientation

    model = _model()
    target = _pose(grasp_orientation(), [0.45, 0.0, 0.12])
    q = model.ik_retry(target, [np.zeros(6), [0, -0.4, 1.3, -0.5, 0, 0]])
    assert q is not None
    tool = model.fk(q)["tool0"]
    pad = tool @ np.array([-0.040, 0.0, -0.030, 1.0])
    tip = tool @ np.array([-0.040, 0.0, -0.105, 1.0])
    assert pad[2] < 0.095   # 3 cm below the flange
    assert 0.005 < tip[2] < 0.035   # fingertips near, but above, the table


def test_joint_error_is_wrapped_across_the_pi_boundary():
    """+pi and -pi are the same pose, so the error there is zero, not 2*pi."""
    assert joint_error(np.array([3.1416]), np.array([-3.1416])) < 0.01
    assert abs(joint_error(np.array([1.0, 0.0]), np.array([0.5, 0.0])) - 0.5) < 1e-9
    assert joint_error(np.array([0.0, 2.0]), np.array([0.0, -2.0])) > 2.0


def test_nearest_joint_branch_avoids_full_turn():
    """+-pi describe the same pose but a full turn apart in joint space."""
    model = _model()
    lower, upper = model.limits
    current = np.zeros(6)
    current[3] = 3.14          # arm parked near the positive limit
    target = current.copy()
    target[3] = -3.14          # same pose, wrong branch coming out of IK

    q = nearest_joint_branch(target, current, lower, upper)
    assert abs(q[3] - 3.14) < 0.01
    assert np.allclose(q[:3], target[:3])

    # Ordinary mid-range targets must be left untouched.
    mid = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert np.allclose(nearest_joint_branch(mid, np.zeros(6), lower, upper), mid)


def test_ik_survives_bad_seeds_and_targets():
    """A joint a hair outside its limit used to make scipy raise.

    scipy's least_squares rejects an infeasible x0 ("x0 is infeasible"), which
    killed the execution node in the middle of an episode, so ik() has to
    sanitise seeds itself.
    """
    model = _model()
    orient = np.diag([1.0, -1.0, -1.0])
    target = _pose(orient, [0.45, 0.0, 0.095])
    bad_seeds = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.5]),   # beyond a joint limit
        np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),  # NaN from a state glitch
        np.array([0.0, -2.8, 2.8, 0.0, 0.0, 3.1416]),  # exactly on the limits
    ]
    q = model.ik_retry(target, bad_seeds, n_random=4)
    assert q is not None
    assert np.all(np.isfinite(q))

    target_nan = target.copy()
    target_nan[0, 3] = np.nan
    assert model.ik(target_nan, np.zeros(6)) is None


def test_ik_viewpoints_and_home():
    model = _model()
    cam_tool = model.camera_in_tool()
    poses = [
        ([0.68, 0.0, 0.62], [0.45, 0.0, 0.02]),
        ([0.45, 0.00, 0.72], [0.45, 0.0, 0.02]),
        ([0.45, -0.28, 0.52], [0.45, 0.0, 0.02]),
        ([0.45, 0.28, 0.52], [0.45, 0.0, 0.02]),
        ([0.18, 0.00, 0.60], [0.45, 0.0, 0.02]),
    ]
    for pos, look in poses:
        cam_target = _pose(look_at_matrix(pos, look), pos)
        tool_target = cam_target @ cam_tool
        q = model.ik_retry(tool_target, [np.zeros(6)], random_seed=7)
        assert q is not None, f"viewpoint IK failed at {pos}"
        # The reconstructed camera must look at the workspace center.
        t_cam = model.fk(q)["camera_link"]
        view_dir = t_cam[:3, :3] @ np.array([1.0, 0, 0])
        expect = np.array(look) - np.array(pos)
        expect /= np.linalg.norm(expect)
        assert np.dot(view_dir, expect) > 0.999


def test_ik_bin_pose():
    model = _model()
    orient = np.diag([1.0, -1.0, -1.0])
    target = _pose(orient, [0.45, -0.34, 0.32])
    q = model.ik_retry(target, [np.zeros(6)], random_seed=7)
    assert q is not None
    t = model.fk(q)["tool0"]
    assert np.linalg.norm(t[:3, 3] - [0.45, -0.34, 0.32]) < 0.02


def test_ik_respects_joint_limits():
    model = _model()
    lower, upper = model.limits
    orient = np.diag([1.0, -1.0, -1.0])
    q = model.ik_retry(
        _pose(orient, [0.45, 0.0, 0.095]), [np.zeros(6)], random_seed=7
    )
    assert q is not None
    assert np.all(q >= lower - 1e-6)
    assert np.all(q <= upper + 1e-6)
