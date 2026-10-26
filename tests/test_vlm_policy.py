import json

import numpy as np

from ag_agent.actions import ActionSpace
from ag_agent.config import load_config
from ag_agent.vlm_policy import (
    VLMPolicy,
    action_from_payload,
    build_messages,
    extract_json,
    summarise_state,
)


def _state():
    state = np.zeros((6, 48, 48), dtype=np.float32)
    state[0, 21:24, 29:33] = 0.04          # one object
    state[0, 30:32, 20:22] = 0.025         # another object
    state[1, :, :] = 0.2
    state[5, 21:24, 29:33] = 1.0
    state[5, 30:32, 20:22] = 1.0
    return state


def _space():
    return ActionSpace(48, 48, n_dirs=8, n_views=4, reach_u=(19, 39), reach_v=(12, 34))


def test_summarise_state_reports_peaks_and_window():
    summary = summarise_state(_state(), _space())
    assert summary["peaks"], "expected at least one peak"
    assert summary["peaks"][0]["height_m"] > 0.02
    assert summary["reachable_window"] == {"u": [19, 39], "v": [12, 34]}
    assert summary["occupied_cells"] > 0
    assert len(summary["coarse_height_cm"]) > 0


def test_build_messages_mentions_the_json_schema():
    messages = build_messages(summarise_state(_state(), _space()))
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "grasp" in messages[1]["content"]
    assert "JSON" in messages[1]["content"]


def test_extract_json_accepts_code_fences():
    payload = extract_json("```json\n{\"kind\": \"grasp\", \"u\": 30, \"v\": 22}\n```")
    assert payload == {"kind": "grasp", "u": 30, "v": 22}
    assert extract_json("no json here") is None


def test_action_validation_respects_the_mask():
    space = _space()
    valid = space.encode_grasp(30, 22)
    mask = np.zeros(space.total, dtype=bool)
    mask[valid] = True
    assert action_from_payload({"kind": "grasp", "u": 30, "v": 22}, space, mask) == valid
    assert action_from_payload({"kind": "grasp", "u": 5, "v": 5}, space, mask) is None
    assert action_from_payload({"kind": "view", "view": 9}, space, mask) is None


class _FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.text


class _FixedFallback:
    def __init__(self, action):
        self.action = action

    def act(self, state, mask):
        return self.action


def test_vlm_policy_uses_a_valid_answer():
    space = _space()
    mask = np.zeros(space.total, dtype=bool)
    expected = space.encode_grasp(30, 22)
    mask[expected] = True
    client = _FakeClient(json.dumps({"kind": "grasp", "u": 30, "v": 22}))
    policy = VLMPolicy(space, {}, client=client, fallback=_FixedFallback(0))
    assert policy.act(_state(), mask) == expected
    assert policy.stats["calls"] == 1
    assert policy.stats["errors"] == 0


def test_vlm_policy_falls_back_on_bad_answer():
    space = _space()
    mask = np.zeros(space.total, dtype=bool)
    mask[space.encode_view(1)] = True
    policy = VLMPolicy(
        space, {}, client=_FakeClient("I cannot help with that."),
        fallback=_FixedFallback(space.encode_view(1)),
    )
    action = policy.act(_state(), mask)
    assert action == space.encode_view(1)
    assert policy.stats["errors"] == 1
    assert policy.stats["fallbacks"] == 1
    assert policy.summary_stats()["calls"] == 1


def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    monkeypatch.setenv("AGENT_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("AGENT_API_MODEL", "local-model")
    config = load_config()
    vlm = config["policy"]["vlm"]
    assert vlm["api_key"] == "secret"
    assert vlm["base_url"] == "http://localhost:8000/v1"
    assert vlm["model"] == "local-model"
