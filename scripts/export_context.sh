#!/usr/bin/env bash
# Bundle the handoff context so it can be uploaded to ChatGPT.
set -euo pipefail

cd "$(dirname "$0")/.."

stamp="$(date +%Y%m%d_%H%M)"
out_dir="context/export-$stamp"
archive="context/active-grasp-context-$stamp.tar.gz"
single_md="context/active-grasp-context-$stamp.md"

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

# A single Markdown file is the easiest thing to attach or paste in ChatGPT
# Work, where a tar.gz may not be previewable.
{
  echo "# Active-Grasp context bundle ($stamp)"
  echo
  echo "> Self-contained handoff. Read this file top to bottom."
  echo
  cat HANDOFF.md
  echo
  echo "---"
  echo
  echo "# Appendix A - README.md"
  echo
  cat README.md
  echo
  echo "---"
  echo
  echo "# Appendix B - config/experiment.yaml"
  echo
  cat config/experiment.yaml
  echo
  echo "---"
  echo
  echo "# Appendix C - docs/09_API策略实验.md"
  echo
  cat "docs/09_API策略实验.md"
  echo
  echo "---"
  echo
  echo "# Appendix D - git log"
  echo
  git log --oneline -30
} > "$single_md"

echo "已生成："
echo "  $single_md   <- 推荐：直接附加/粘贴到 ChatGPT Work"
echo "  $archive     <- 完整文件包（GitHub 不可用时）"
echo "然后粘贴 HANDOFF.md 第 9 节的开场提示。"
