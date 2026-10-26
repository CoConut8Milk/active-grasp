"""API-backed policy adapter (any OpenAI-compatible chat endpoint).

The policy receives the same 6x48x48 state tensor as the DDQN agent, turns it
into a compact textual description (optionally plus a rendered height-map
image), asks the model for exactly one action as JSON, validates that action
against the legal-action mask and falls back to the greedy policy on any
error.  No third-party HTTP library is required - only the standard library.
"""

import base64
import io
import json

import os
import time
import urllib.error
import urllib.request

import numpy as np

from ag_agent.baselines import GreedyPolicy


# --------------------------------------------------------------------------
# state -> prompt
# --------------------------------------------------------------------------
def _reach_window(action_space):
    reach_u = getattr(action_space, "reach_u", None)
    reach_v = getattr(action_space, "reach_v", None)
    if reach_u is None:
        b = getattr(action_space, "border", 0)
        reach_u = (b, action_space.width - 1 - b)
    if reach_v is None:
        b = getattr(action_space, "border", 0)
        reach_v = (b, action_space.height - 1 - b)
    return reach_u, reach_v


def summarise_state(state, action_space, top=8, grid=6):
    """Compact, token-efficient description of the fused perception state."""
    state = np.asarray(state, dtype=np.float32)
    height = state[0]
    uncertainty = state[1]
    occupied = state[5] > 0.5
    rows, cols = height.shape
    reach_u, reach_v = _reach_window(action_space)

    window = np.zeros_like(occupied)
    window[reach_v[0]: reach_v[1] + 1, reach_u[0]: reach_u[1] + 1] = True

    cells = np.argwhere(occupied & window)
    peaks = []
    if len(cells):
        order = np.argsort([height[v, u] for v, u in cells])[::-1][:top]
        for index in order:
            v, u = cells[index]
            peaks.append(
                {
                    "u": int(u),
                    "v": int(v),
                    "height_m": round(float(height[v, u]), 3),
                    "uncertainty": round(float(uncertainty[v, u]), 2),
                }
            )

    # Coarse height map: mean height per block, in centimetres.
    gy = max(1, rows // grid)
    gx = max(1, cols // grid)
    coarse = []
    for by in range(0, rows - gy + 1, gy):
        line = []
        for bx in range(0, cols - gx + 1, gx):
            line.append(round(float(np.mean(height[by: by + gy, bx: bx + gx])) * 100, 1))
        coarse.append(line)

    reach_unc = uncertainty[window]
    summary = {
        "grid": {"height": int(rows), "width": int(cols)},
        "reachable_window": {"u": list(map(int, reach_u)), "v": list(map(int, reach_v))},
        "occupied_cells": int(np.count_nonzero(occupied & window)),
        "unknown_cells_in_window": int(np.count_nonzero(~occupied & window)),
        "mean_uncertainty_in_window": (
            round(float(np.mean(reach_unc)), 3) if reach_unc.size else None
        ),
        "peaks": peaks,
        "coarse_height_cm": coarse,
        "views_available": list(range(int(getattr(action_space, "n_views", 0)))),
    }
    return summary


SYSTEM_PROMPT = (
    "You control a 6-DoF tabletop robot arm with a parallel-jaw gripper and an "
    "eye-in-hand RGB-D camera. The arm must clear cluttered tabletop objects "
    "into a bin. You see a fused height map: higher cells are objects, "
    "uncertainty encodes occlusion/holes/staleness.\n"
    "Choose exactly one action and answer with a single JSON object, no prose, "
    "no markdown."
)


def build_messages(summary, include_image=None):
    """Build chat messages. `include_image` is an optional base64 data URI."""
    instructions = (
        "Actions (all indices are height-map cells):\n"
        "- {\"kind\": \"grasp\", \"u\": int, \"v\": int}  vertical grasp\n"
        "- {\"kind\": \"push\", \"u\": int, \"v\": int, \"dir\": 0-7}  push, "
        "dir 0=+x, 2=+y, 4=-x, 6=-y\n"
        "- {\"kind\": \"view\", \"view\": int}  move to another camera view\n"
        "Rules: prefer grasping a well-localised peak; consider a view when "
        "mean uncertainty or unknown cells are high; push when clutter blocks "
        "a grasp. Only pick u,v inside reachable_window. Answer with JSON only."
    )
    user_text = (
        "Scene summary (JSON):\n"
        + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
        + instructions
        + "\nExample: {\"kind\":\"grasp\",\"u\":30,\"v\":22}"
    )
    if include_image:
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": include_image}},
        ]
    else:
        user_content = user_text
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------
# response -> action
# --------------------------------------------------------------------------
def extract_json(text):
    """Extract the first JSON object from a model answer (tolerates fences)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(cleaned[start: index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def action_from_payload(payload, action_space, mask):
    """Validate a model answer and convert it to a flat action index."""
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind", "")).lower()
    try:
        if kind == "grasp":
            u, v = int(payload["u"]), int(payload["v"])
            if not (0 <= u < action_space.width and 0 <= v < action_space.height):
                return None
            action = action_space.encode_grasp(u, v)
        elif kind == "push":
            u, v = int(payload["u"]), int(payload["v"])
            direction = int(payload["dir"]) % action_space.n_dirs
            if not (0 <= u < action_space.width and 0 <= v < action_space.height):
                return None
            action = action_space.encode_push(u, v, direction)
        elif kind == "view":
            view = int(payload.get("view", payload.get("view_id", -1)))
            if not (0 <= view < action_space.n_views):
                return None
            action = action_space.encode_view(view)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if mask is not None and not bool(mask[action]):
        return None
    return int(action)


def render_heightmap(state, max_side=96):
    """Optional PNG data URI of the height map (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    height = np.asarray(state[0], dtype=np.float32)
    finite = height[np.isfinite(height)]
    if finite.size == 0:
        return None
    top = max(float(finite.max()), 0.05)
    image = np.clip(height / top, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    pil = Image.fromarray(image, mode="L").resize(
        (max_side, max_side), Image.NEAREST
    )
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------
class ChatClient:
    """Minimal OpenAI-compatible /chat/completions client (stdlib only)."""

    def __init__(self, base_url, api_key, model, temperature=0.0,
                 max_tokens=300, timeout_s=30.0, retries=2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)

    def chat(self, messages):
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError,
                    json.JSONDecodeError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise RuntimeError(f"chat API failed: {last_error}")


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------
class VLMPolicy:
    """API policy with legal-action validation and a greedy fallback."""

    def __init__(self, action_space, vlm_config, client=None, fallback=None,
                 logger=None):
        self.action_space = action_space
        self.config = dict(vlm_config or {})
        self.client = client or ChatClient(
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
            api_key=self.config.get("api_key") or os.environ.get("AGENT_API_KEY", ""),
            model=self.config.get("model", "gpt-4o-mini"),
            temperature=self.config.get("temperature", 0.0),
            max_tokens=self.config.get("max_tokens", 300),
            timeout_s=self.config.get("timeout_s", 30.0),
            retries=self.config.get("retries", 2),
        )
        self.fallback = fallback if fallback is not None else GreedyPolicy(action_space)
        self.logger = logger
        self.include_image = bool(self.config.get("include_image", False))
        self.image_max_side = int(self.config.get("image_max_side", 96))
        self.stats = {"calls": 0, "errors": 0, "fallbacks": 0,
                      "latency_s": [], "last_error": None}
        self.last_summary = None
        self.last_response = None

    def _log(self, level, message):
        if self.logger is not None:
            getattr(self.logger, level, self.logger.info)(message)

    def act(self, state, mask):
        self.last_summary = summarise_state(state, self.action_space)
        image = render_heightmap(state, self.image_max_side) if self.include_image else None
        messages = build_messages(self.last_summary, include_image=image)
        self.stats["calls"] += 1
        started = time.monotonic()
        try:
            response = self.client.chat(messages)
            self.last_response = response
            payload = extract_json(response)
            action = action_from_payload(payload, self.action_space, mask)
            if action is None:
                raise ValueError(f"invalid action payload: {payload!r}")
            return action
        except Exception as exc:  # noqa: BLE001 - any API failure must not stop the demo
            self.stats["errors"] += 1
            self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
            self._log("warning", f"API policy failed ({exc}); using fallback")
            if self.fallback is None:
                raise
            self.stats["fallbacks"] += 1
            return int(self.fallback.act(state, mask))
        finally:
            self.stats["latency_s"].append(time.monotonic() - started)

    def observe_result(self, action, success):
        if hasattr(self.fallback, "observe_result"):
            self.fallback.observe_result(action, success)

    def summary_stats(self):
        latency = self.stats["latency_s"]
        return {
            "calls": self.stats["calls"],
            "errors": self.stats["errors"],
            "fallbacks": self.stats["fallbacks"],
            "mean_latency_s": round(float(np.mean(latency)), 3) if latency else 0.0,
            "last_error": self.stats["last_error"],
        }
