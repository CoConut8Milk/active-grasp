import numpy as np

from ag_agent.actions import ActionSpace


def test_encode_decode_round_trip():
    space = ActionSpace(48, 48, n_dirs=8, n_views=4)
    for action in range(space.total):
        kind, *payload = space.decode(action)
        if kind == "grasp":
            assert space.encode_grasp(*payload) == action
        elif kind == "push":
            assert space.encode_push(*payload) == action
        else:
            assert space.encode_view(*payload) == action


def test_valid_mask():
    space = ActionSpace(4, 4, n_dirs=2, n_views=2)
    state = np.zeros((6, 4, 4), dtype=np.float32)
    state[5, 1, 1] = 1.0
    mask = space.valid_mask(state)
    assert mask.sum() == 1 + 2 + 2  # one grasp cell, 2 push dirs, 2 views
    assert mask[space.encode_grasp(1, 1)]
    assert mask[space.encode_push(1, 1, 0)]
    assert mask[space.encode_push(1, 1, 1)]
    assert mask[space.encode_view(0)]
    assert not mask[space.encode_grasp(0, 0)]


def test_valid_mask_excludes_unreliable_border():
    """The depth image border is noise, and the agent must never grasp there."""
    space = ActionSpace(48, 48, n_dirs=8, n_views=4)
    state = np.zeros((6, 48, 48), dtype=np.float32)
    state[5, 0, 47] = 1.0    # permanent corner artefact seen in gazebo
    state[5, 24, 24] = 1.0   # a real object in the middle of the table
    mask = space.valid_mask(state)
    assert not mask[space.encode_grasp(47, 0)]
    assert mask[space.encode_grasp(24, 24)]


def test_blob_centre_snaps_edge_pick_to_object_centre():
    from ag_agent.actions import blob_centre

    occupied = np.zeros((48, 48), dtype=bool)
    occupied[28:33, 18:24] = True   # a 5x6 blob, tallest cell on its corner
    u, v = blob_centre(occupied, 18, 28)
    assert 20 <= u <= 21
    assert 30 <= v <= 31
    # Empty neighbourhood keeps the requested cell.
    assert blob_centre(np.zeros((48, 48), dtype=bool), 5, 7) == (5, 7)

