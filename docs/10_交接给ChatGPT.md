# 10 把本项目的上下文交给 ChatGPT

本地 Codex 对话**不会自动同步**到 ChatGPT：两者是不同产品、不同模型、
不同会话存储。要让 ChatGPT 接手，需要给它一份「上下文包」或让它读 GitHub。

## 方式 A（推荐）：上传上下文压缩包

在 Ubuntu 或 Windows 的仓库根目录执行：

```bash
bash scripts/export_context.sh
```

会生成：

```
context/active-grasp-context-<时间戳>.md      ← 推荐
context/active-grasp-context-<时间戳>.tar.gz  ← 完整包
```

然后在 ChatGPT 网页端新建对话，**附加这个 `.md` 文件**（或压缩包），并粘贴
`HANDOFF.md` 第 9 节的开场提示。

## ChatGPT Work 专用说明

ChatGPT Work 是**云端任务**，同样读不到你本地的 WSL 文件和本地 Codex
会话。可用方式只有两种：

1. **新 Work 任务 + 附件**：把 `active-grasp-context-<时间戳>.md` 直接
   附加到 Work 任务里，粘贴开场提示。单文件最容易读取。
2. **新 Work 任务 + GitHub 连接**：把仓库连接到该 Work 项目，让它读
   `HANDOFF.md`。私有仓库必须用连接方式，不能只给链接。

注意分工：Work 能改代码、读日志、写文档，但**不能运行你本地的 Gazebo**。
仿真仍然在实验室机器上跑，把终端日志贴给 Work 即可。

## 方式 B：让 ChatGPT 直接读 GitHub

1. 确认最新代码已推送：`git push origin main`；
2. 在 ChatGPT 中启用 GitHub 连接（或把仓库设为公开），让它读取
   `HANDOFF.md`、`README.md`、`config/experiment.yaml`；
3. 粘贴同样的开场提示。

私有仓库必须用「连接 GitHub」的方式，不能只给链接。

## 方式 C：纯文本粘贴

把 `HANDOFF.md` 全文粘贴到 ChatGPT 对话里（约几十 KB，通常可以）。
如果超出长度，按「现状 / 环境 / 运行命令 / 已知问题 / 下一步」分几条发送。

## 不要上传什么

- `.env`、任何 `sk-…` key、`results/` 里的原始日志（除非你需要它分析）；
- 如果必须给日志，先检查是否含有密钥或隐私路径。

## 上传后怎么继续

把新的 `type (x).txt` 终端日志直接发给 ChatGPT，并说明：

> 这是最新一次运行的完整日志，请按 HANDOFF.md 第 6 节逐条核对，
> 给出根因、修改文件和验证命令。
