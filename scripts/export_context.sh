#!/usr/bin/env bash
# Bundle the handoff context so it can be uploaded to ChatGPT.
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="$(date +%Y%m%d_%H%M)"
out_dir="context/export-$stamp"
archive="context/active-grasp-context-$stamp.tar.gz"

mkdir -p "$out_dir"
cp HANDOFF.md README.md "$out_dir"/
cp -r config "$out_dir"/
cp docs/09_API策略实验.md docs/10_交接给ChatGPT.md "$out_dir"/
git log --oneline -30 > "$out_dir/git-log.txt"
git status -sb > "$out_dir/git-status.txt" 2>/dev/null || true

if [ -f .env ]; then
  echo "⚠  检测到 .env：不会打包（密钥不能上传）。"
fi

tar -czf "$archive" -C context "export-$stamp"
rm -rf "$out_dir"

echo "已生成：$archive"
echo "把它附加到 ChatGPT 对话，并粘贴 HANDOFF.md 第 9 节的开场提示。"
