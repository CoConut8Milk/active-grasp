# `config/` — 实验配置

| 文件 | 用途 |
|---|---|
| `experiment.yaml` | 唯一需要修改的实验入口：场景规模、策略类型（greedy / checkpoint / vlm）、API 参数、结果目录 |

## 配置优先级

1. 命令行参数（`--policy`、`--objects`、`--config`）
2. 环境变量（`AGENT_API_KEY`、`AGENT_API_BASE`、`AGENT_API_MODEL`）
3. `experiment.yaml`
4. 代码内置默认值

## 密钥

**不要把 API Key 写进 `experiment.yaml`。** 在仓库根目录创建 `.env`（已被
`.gitignore` 忽略），内容参考 `.env.example`：

```bash
export AGENT_API_KEY="sk-..."
export AGENT_API_BASE="https://api.openai.com/v1"   # 可选
export AGENT_API_MODEL="gpt-4o-mini"                # 可选
source .env
```

运行 `bash scripts/check_api.sh` 可以先验证 key、endpoint 和模型名是否可用。
