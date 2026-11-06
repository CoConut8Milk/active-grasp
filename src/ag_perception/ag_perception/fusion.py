"""Pure-numpy multi-view heightmap fusion.

This module contains no ROS types so it can be unit-tested anywhere.

The world model is a top-down grid over the table workspace. Each
observation back-projects a depth image into the base frame and produces a
fresh "snapshot" of the scene. Two snapshots are combined into the state:

  - the home snapshot (default oblique view), refreshed on every action;
  - the auxiliary snapshot (last chosen viewpoint), kept until the next
    manipulation invalidates it.

An uncertainty map scores every cell as a mix of

  - hole:   never seen in the current world model,
  - stale:  not observed recently (information age),
  - edge:   strong height discontinuity, the signature of an occlusion
            boundary.

This is the signal the agent uses to decide whether moving the camera to
another viewpoint is worth the cost.
"""

import numpy as np


def pose_to_matrix(translation, rotation_xyzw):
    """Build a 4x4 cam->base transform from translation + quaternion (x,y,z,w)."""
    x, y, z, w = rotation_xyzw
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = translation
    return t


def backproject(depth, k_matrix, t_base_cam, u_sign=1.0, v_sign=1.0):
    """Convert a depth image into 3D points in the base frame.

    The Gazebo Classic camera looks along +x of its link, so the ray for
    pixel (u, v) is:
        d = (1, (u-cx)/fx * u_sign, (v-cy)/fy * v_sign)
        p_cam = depth * d
    followed by the cam->base rigid transform.

    u_sign/v_sign absorb the possible image mirroring of the simulator; they
    are determined once by auto_calibrate().
    """
    height, width = depth.shape
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]

    vv, uu = np.mgrid[0:height, 0:width]
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05)

    x_c = z
    y_c = (uu - cx) * z / fx * u_sign
    z_c = (vv - cy) * z / fy * v_sign
    ones = np.ones_like(z)
    p_c = np.stack([x_c, y_c, z_c, ones], axis=-1)
    p_b = np.einsum("ij,hwj->hwi", t_base_cam, p_c)
    return p_b[..., :3], valid


def auto_calibrate_scores(depth, k_matrix, t_base_cam, max_depth=1.1,
                          max_samples=3000):
    """Flatness score (z variance) for all four mirroring hypotheses.

    A lower score means the points back-project closer to one horizontal
    plane.  Returning all four lets the caller judge whether the view can
    actually distinguish the signs (a straight top-down view cannot).
    """
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < max_depth)
    idx = np.argwhere(valid)
    if idx.shape[0] < 200:
        raise RuntimeError("calibration failed: not enough valid depth pixels")
    if idx.shape[0] > max_samples:
        idx = idx[np.random.default_rng(0).choice(idx.shape[0], max_samples, replace=False)]

    z = depth[idx[:, 0], idx[:, 1]]
    u = idx[:, 1].astype(np.float64)
    v = idx[:, 0].astype(np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]

    scores = {}
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            p = np.stack(
                [
                    z,
                    (u - cx) * z / fx * su,
                    (v - cy) * z / fy * sv,
                    np.ones_like(z),
                ],
                axis=-1,
            )
            p_b = np.einsum("ij,nj->ni", t_base_cam, p)
            scores[(su, sv)] = float(np.var(p_b[:, 2]))
    return scores


def auto_calibrate(depth, k_matrix, t_base_cam, max_depth=1.1, max_samples=3000):
    """Recover the best (u_sign, v_sign) from a flat table frame.

    Note: for a level table the horizontal sign is mathematically
    unobservable (mirroring keeps the plane flat); see
    ``choose_v_sign`` for the vertical decision.
    """
    scores = auto_calibrate_scores(
        depth, k_matrix, t_base_cam,
        max_depth=max_depth, max_samples=max_samples,
    )
    return min(scores, key=scores.get)


def choose_v_sign(scores, fallback=-1.0, ratio=0.25):
    """Decide the vertical image sign only when the data can see it.

    ``scores`` maps (su, sv) -> flatness variance.  Because u is
    unobservable for a level table, compare the *best* score achievable for
    each v.  If the two are within ``ratio`` of each other the view is
    degenerate (e.g. perfectly top-down) and the configured ``fallback`` is
    returned instead of a coin flip.

    Returns ``(v_sign, confident, best_by_v)``.
    """
    best_by_v = {
        1.0: min(scores[(1.0, 1.0)], scores[(-1.0, 1.0)]),
        -1.0: min(scores[(1.0, -1.0)], scores[(-1.0, -1.0)]),
    }
    best_v = min(best_by_v, key=best_by_v.get)
    other = -best_v
    # Require a real margin: when both variances are ~0 (perfect top-down
    # view) the ratio test alone would call the tie "confident".
    margin = best_by_v[other] - best_by_v[best_v]
    confident = (
        margin > 1e-6
        and best_by_v[best_v] < ratio * max(best_by_v[other], 1e-12)
    )
    return (best_v if confident else float(fallback)), confident, best_by_v


def make_snapshot(points, valid, color, height, width, x_min, x_max, y_min, y_max,
                  z_min=0.005, z_max=0.35):
    """Project 3D points into the grid; keep max height and its color per cell."""
    grid_h = np.full((height, width), -1.0, dtype=np.float32)
    grid_color = np.zeros((height, width, 3), dtype=np.uint8)
    grid_valid = np.zeros((height, width), dtype=bool)

    xs, ys, zs = points[..., 0], points[..., 1], points[..., 2]
    sel = (
        valid
        & (zs >= z_min)
        & (zs <= z_max)
        & (xs >= x_min)
        & (xs < x_max)
        & (ys >= y_min)
        & (ys < y_max)
    )
    res = (x_max - x_min) / width
    ci = ((xs[sel] - x_min) / res).astype(np.int32)
    cj = ((y_max - ys[sel]) / res).astype(np.int32)
    ok = (ci >= 0) & (ci < width) & (cj >= 0) & (cj < height)
    ci, cj = ci[ok], cj[ok]
    zvals = zs[sel][ok]
    colors = color[sel][ok]
    flat = cj * width + ci

    np.maximum.at(grid_h.reshape(-1), flat, zvals)
    grid_valid.reshape(-1)[flat] = True
    # Writing points sorted by ascending height leaves the color of the
    # highest point in each cell last.
    order = np.argsort(zvals, kind="stable")
    grid_color.reshape(-1, 3)[flat[order]] = colors[order]
    return {"height": grid_h, "color": grid_color, "valid": grid_valid}


def _neighbor_edge(height, valid):
    """Height discontinuity score per cell: max |dh| over valid 4-neighbors."""
    padded_h = np.full(
        (height.shape[0] + 2, height.shape[1] + 2), -1.0, dtype=np.float32
    )
    padded_v = np.zeros_like(padded_h, dtype=bool)
    padded_h[1:-1, 1:-1] = height
    padded_v[1:-1, 1:-1] = valid

    neighbors = []
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nh = padded_h[1 + di: height.shape[0] + 1 + di,
                     1 + dj: height.shape[1] + 1 + dj]
        nv = padded_v[1 + di: height.shape[0] + 1 + di,
                     1 + dj: height.shape[1] + 1 + dj]
        neighbors.append((nh, nv))

    edge = np.zeros(height.shape, dtype=np.float32)
    for nh, nv in neighbors:
        both = valid & nv
        diff = np.abs(height - nh)
        edge[both] = np.maximum(edge[both], diff[both])
    return np.minimum(edge / 0.03, 1.0)


class HeightmapFusion:
    def __init__(self, height, width, x_min, x_max, y_min, y_max,
                 z_max=0.35, max_age=8, w_hole=0.30, w_age=0.35, w_edge=0.35):
        self.height = int(height)
        self.width = int(width)
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_max = z_max
        self.max_age = max_age
        self.w_hole, self.w_age, self.w_edge = w_hole, w_age, w_edge
        self._home = None
        self._aux = None
        self._age = np.full((self.height, self.width), max_age, dtype=np.float32)
        self._step = 0

    def reset(self):
        self._home = None
        self._aux = None
        self._age[...] = self.max_age
        self._step = 0

    def clear_aux(self):
        self._aux = None

    def build_snapshot(self, points, valid, color):
        return make_snapshot(
            points, valid, color, self.height, self.width,
            self.x_min, self.x_max, self.y_min, self.y_max, z_max=self.z_max,
        )

    def observe(self, snapshot, auxiliary):
        if auxiliary:
            self._aux = snapshot
        else:
            self._home = snapshot
        seen = snapshot["height"] >= 0
        self._age[seen] = 0
        self._age[~seen] += 1
        self._age = np.minimum(self._age, self.max_age)
        self._step += 1
        return self.state()

    def state(self):
        grid_h = np.zeros((self.height, self.width), dtype=np.float32)
        grid_color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        valid = np.zeros((self.height, self.width), dtype=bool)

        for snap in (self._home, self._aux):
            if snap is None:
                continue
            v = snap["height"] >= 0
            take = (~valid | (snap["height"] > grid_h)) & v
            grid_h[take] = snap["height"][take]
            grid_color[take] = snap["color"][take]
            valid |= v

        hole = (~valid).astype(np.float32)
        stale = self._age / self.max_age
        edge = _neighbor_edge(grid_h, valid)
        uncertainty = self.w_hole * hole + self.w_age * stale + self.w_edge * edge
        return grid_h, np.clip(uncertainty, 0.0, 1.0), grid_color, valid
