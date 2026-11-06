import numpy as np
import pytest

from ag_perception.fusion import (
    HeightmapFusion,
    auto_calibrate,
    auto_calibrate_scores,
    backproject,
    choose_v_sign,
    make_snapshot,
    pose_to_matrix,
)


def _pinhole_k(width=320, height=240, fov=1.0):
    f = width / (2 * np.tan(fov / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


def _flat_table_depth(width=320, height=240, u_sign=1.0, v_sign=-1.0):
    """Synthetic depth image of the z=0 plane from a tilted camera."""
    k = _pinhole_k(width, height)
    # Camera off to the side so the up-cross in look_at is well defined.
    position = np.array([0.5, 0.25, 0.5])
    target = np.array([0.5, 0.0, 0.0])
    x = target - position
    x /= np.linalg.norm(x)
    y = np.cross([0, 0, 1], x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    r = np.column_stack([x, y, z])

    vv, uu = np.mgrid[0:height, 0:width]
    dirs_cam = np.stack(
        [
            np.ones_like(uu, dtype=float),
            (uu - k[0, 2]) / k[0, 0] * u_sign,
            (vv - k[1, 2]) / k[1, 1] * v_sign,
        ],
        axis=-1,
    )
    dirs_world = dirs_cam @ r.T
    depth = np.where(
        dirs_world[..., 2] < -1e-6,
        -position[2] / dirs_world[..., 2],
        np.nan,
    )
    return depth, k, np.vstack([np.hstack([r, position[:, None]]), [0, 0, 0, 1]])


def test_backproject_flat_table_round_trip():
    depth, k, t = _flat_table_depth()
    # Only the sign combination that matches the image generation convention
    # reconstructs the plane; the other three are intentionally mirrored.
    points, valid = backproject(depth, k, t, u_sign=1.0, v_sign=-1.0)
    assert np.allclose(points[valid][:, 2], 0.0, atol=1e-6)


def test_auto_calibrate_recovers_signs():
    depth, k, t = _flat_table_depth(u_sign=1.0, v_sign=-1.0)
    su, sv = auto_calibrate(depth, k, t, max_samples=1500)
    assert (su, sv) == (1.0, -1.0)


def test_flat_table_cannot_observe_horizontal_sign():
    """The mirrored view of a level table is level as well.

    This is why the perception node takes the horizontal sign from the camera
    convention instead of from the flatness estimate: both u choices fit the
    plane equally well, so the estimate would otherwise be a coin flip.
    """
    depth, k, t = _flat_table_depth(u_sign=1.0, v_sign=-1.0)

    def height_variance(su, sv):
        points, valid = backproject(depth, k, t, u_sign=su, v_sign=sv)
        return float(np.var(points[valid][:, 2]))

    assert height_variance(1.0, -1.0) == pytest.approx(
        height_variance(-1.0, -1.0), abs=1e-9
    )
    # The vertical sign, on the other hand, is clearly observable.
    assert height_variance(1.0, -1.0) < height_variance(1.0, 1.0)
    _, sv = auto_calibrate(depth, k, t, max_samples=1500)
    assert sv == -1.0


def test_choose_v_sign_keeps_the_fallback_when_topdown():
    # A perfectly top-down view fits both vertical mirrors equally well.
    scores = {
        (1.0, 1.0): 0.0, (-1.0, 1.0): 0.0,
        (1.0, -1.0): 0.0, (-1.0, -1.0): 0.0,
    }
    sign, confident, _ = choose_v_sign(scores, fallback=-1.0)
    assert sign == -1.0 and not confident


def test_choose_v_sign_trusts_a_clear_oblique_view():
    scores = {
        (1.0, 1.0): 0.020, (-1.0, 1.0): 0.020,
        (1.0, -1.0): 0.0001, (-1.0, -1.0): 0.0001,
    }
    sign, confident, _ = choose_v_sign(scores, fallback=1.0)
    assert sign == -1.0 and confident


def test_auto_calibrate_scores_returns_four_hypotheses():
    depth, k, t = _flat_table_depth(u_sign=1.0, v_sign=-1.0)
    scores = auto_calibrate_scores(depth, k, t, max_samples=800)
    assert set(scores) == {(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)}
    assert scores[(1.0, -1.0)] == pytest.approx(0.0, abs=1e-9)


def test_snapshot_and_uncertainty():
    fusion = HeightmapFusion(24, 24, 0.0, 0.48, -0.24, 0.24)
    # Synthetic points: flat table everywhere plus a 0.1 m box in the middle.
    xs = np.linspace(0.01, 0.47, 24)
    ys = np.linspace(-0.23, 0.23, 24)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.where((np.abs(xs - 0.24) < 0.06) & (np.abs(ys) < 0.06), 0.10, 0.01)
    points = np.stack([xs, ys, zs], axis=-1)
    valid = np.ones(xs.shape, dtype=bool)
    color = np.zeros((*xs.shape, 3), dtype=np.uint8)
    color[..., 0] = 200

    snap = fusion.build_snapshot(points, valid, color)
    height, uncertainty, color_map, mask = fusion.observe(snap, auxiliary=False)

    center = height[12, 12]
    assert 0.09 < center <= 0.10
    assert mask[12, 12]
    assert np.all(uncertainty >= 0.0) and np.all(uncertainty <= 1.0)
    # Edge cells (box boundary) carry the highest uncertainty.
    assert uncertainty[12, 12] < np.max(uncertainty)
    # Everything was just observed: no stale cells.
    assert np.max(uncertainty) < 0.75


def test_aux_view_fills_holes():
    fusion = HeightmapFusion(12, 12, 0.0, 0.48, -0.24, 0.24)
    # Home observation sees only the left half of the grid.
    xs = np.linspace(0.0, 0.23, 12)
    ys = np.linspace(-0.23, 0.23, 12)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.full_like(xs, 0.01)
    points = np.stack([xs, ys, zs], axis=-1)
    valid = np.ones_like(xs, dtype=bool)
    color = np.zeros((*xs.shape, 3), dtype=np.uint8)
    home = fusion.build_snapshot(points, valid, color)
    h1, u1, _, valid1 = fusion.observe(home, auxiliary=False)
    holes_before = float((~valid1).mean())

    # Auxiliary view sees the right half.
    xs = np.linspace(0.24, 0.47, 12)
    ys2 = np.linspace(-0.23, 0.23, 12)
    xs, ys2 = np.meshgrid(xs, ys2)
    zs2 = np.full_like(xs, 0.01)
    points = np.stack([xs, ys2, zs2], axis=-1)
    aux = fusion.build_snapshot(points, valid, color)
    h2, u2, _, valid2 = fusion.observe(aux, auxiliary=True)

    assert float((~valid2).mean()) < holes_before
    assert h2[6, 9] > 0.005  # the right half is now visible
    assert float(np.mean(u2)) < float(np.mean(u1))


def test_pose_to_matrix_is_rigid():
    t = pose_to_matrix([0.1, 0.2, 0.3], [0, 0, 0, 1])
    assert np.allclose(t[:3, :3], np.eye(3))
    assert np.allclose(t[:3, 3], [0.1, 0.2, 0.3])
    q = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    t = pose_to_matrix([0, 0, 0], q)
    assert np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-6)
