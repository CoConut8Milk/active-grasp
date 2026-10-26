#!/usr/bin/env bash
# Run the demo with the API-backed policy (any OpenAI-compatible endpoint).
set -euo pipefail

cd "$(dirname "$0")/.."

# ROS 2 Humble's setup.bash conflicts with `set -u`.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || true
if [ -f .env ]; then
  source .env
fi
set -u

if [ -z "${AGENT_API_KEY:-}" ]; then
  echo "⚠  未设置 AGENT_API_KEY。"
  echo "   1) cp .env.example .env"
  echo "   2) 编辑 .env，填入 key（以及可选的 base_url / model）"
  echo "   3) 重新运行本脚本"
  exit 1
fi

if pgrep -x gzserver >/dev/null 2>&1; then
  echo "⚠  检测到已有 gzserver 在运行（上一次仿真没退干净）。"
  echo "   请先执行： pkill -f gzserver; pkill -f gzclient"
  exit 1
fi

exec ros2 launch ag_agent demo.launch.py \
  policy:=vlm \
  config:="$(pwd)/config/experiment.yaml"
