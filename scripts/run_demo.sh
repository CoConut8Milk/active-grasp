#!/usr/bin/env bash
# Realtime GUI demo with the greedy baseline.
set -euo pipefail

cd "$(dirname "$0")/.."

# ROS 2 Humble 的 setup.bash 与 set -u 冲突，source 期间临时关闭 nounset。
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || true
set -u

# 上次仿真若没有退干净，新 gzserver 会因为端口被占用直接退出（exit 255），
# 而控制器/传感器会悄悄连到那个旧实例上：手臂不动、夹爪不响应、TF 全错。
# 与其跑到一半才发现，不如在这里拦住。
if pgrep -x gzserver >/dev/null 2>&1; then
  echo "⚠  检测到已有 gzserver 在运行（很可能是上一次仿真没退出）。"
  echo "   请先执行："
  echo "     pkill -f gzserver; pkill -f gzclient"
  echo "   等 2 秒后再运行本脚本。"
  exit 1
fi

exec ros2 launch ag_agent demo.launch.py
