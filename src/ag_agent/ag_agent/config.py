"""Experiment configuration loading.

Lookup order (later wins):

    defaults  <  config/experiment.yaml  <  environment variables  <  CLI

Secrets are never stored in the YAML: the file only names the environment
variable that holds the API key.
"""

import os
import pathlib
from copy import deepcopy

import yaml


DEFAULTS = {
    "scene": {"objects": 5, "max_steps": 25},
    "policy": {
        "kind": "greedy",
        "checkpoint": "",
        "vlm": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env": "AGENT_API_KEY",
            "temperature": 0.0,
            "max_tokens": 300,
            "timeout_s": 30.0,
            "retries": 2,
            "include_image": False,
            "image_max_side": 96,
        },
    },
    "logging": {"out_dir": "results/api_runs", "save_jsonl": True},
    "safety": {"fallback": "greedy"},
}


def find_repo_root(start=None):
    """Walk upwards until a directory containing config/experiment.yaml."""
    here = pathlib.Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "experiment.yaml").is_file():
            return candidate
    return pathlib.Path.cwd()


def default_config_path():
    return find_repo_root() / "config" / "experiment.yaml"


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_env_overrides(config):
    """Environment variables override endpoint/model; the key is env-only."""
    vlm = config["policy"]["vlm"]
    key_env = vlm.get("api_key_env") or "AGENT_API_KEY"
    api_key = os.environ.get(key_env) or os.environ.get("AGENT_API_KEY")
    if api_key:
        vlm["api_key"] = api_key

    base = os.environ.get("AGENT_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if base:
        vlm["base_url"] = base
    model = os.environ.get("AGENT_API_MODEL") or os.environ.get("OPENAI_MODEL")
    if model:
        vlm["model"] = model
    return config


def load_config(path=None, overrides=None):
    """Load experiment.yaml (if present) merged over the built-in defaults."""
    config = deepcopy(DEFAULTS)
    config_path = pathlib.Path(path) if path else default_config_path()
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(config, loaded)
    config = apply_env_overrides(config)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def redacted(config):
    """Copy of the config with credentials removed (safe to log)."""
    safe = deepcopy(config)
    vlm = safe.get("policy", {}).get("vlm", {})
    if vlm.get("api_key"):
        vlm["api_key"] = "***"
    return safe
