#!/usr/bin/env bash
# Verify AGENT_API_KEY / AGENT_API_BASE / AGENT_API_MODEL with one tiny call.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set +u
  source .env
  set -u
fi

if [ -z "${AGENT_API_KEY:-}" ]; then
  echo "⚠  未设置 AGENT_API_KEY（先 cp .env.example .env 并填写）"
  exit 1
fi

python3 - <<'PY'
import json
import os
import urllib.request

base = os.environ.get("AGENT_API_BASE", "https://api.openai.com/v1").rstrip("/")
model = os.environ.get("AGENT_API_MODEL", "gpt-4o-mini")
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    "max_tokens": 5,
}).encode()
request = urllib.request.Request(
    f"{base}/chat/completions",
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['AGENT_API_KEY']}",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    print("API OK:", payload["choices"][0]["message"]["content"].strip())
except Exception as exc:  # noqa: BLE001 - report and fail
    print(f"API FAILED: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
PY
