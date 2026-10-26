# 09 用 API 模型跑策略实验

这一章把"新 API"接进实验：任何 **OpenAI 兼容的 `/chat/completions`**
端点都可以作为策略大脑，在同一个 Gazebo 场景里和贪心基线、DDQN 做对照。

## 1. 它做什么

`ag_agent/vlm_policy.py` 把感知状态（48×48 高度图 + 不确定度 + 占用掩码）
压缩成一段 JSON 摘要，问模型"下一步做什么"，模型必须只回一个 JSON 动作：

```json
{"kind": "grasp", "u": 30, "v": 22}
{"kind": "push", "u": 30, "v": 22, "dir": 6}
{"kind": "view", "view": 2}
```

- 动作会经过**合法性校验**（范围、方向、动作掩码、可达窗口）；
- 任何失败（超时、401、返回非 JSON、动作非法）都会**自动回退到贪心策略**，
  并记录 `errors` / `fallbacks`，不会中断实验；
- 每一步都会写入 JSONL，包含动作、奖励、清理数、API 延迟和状态摘要。

## 2. 支持的后端

| 服务 | base_url 示例 |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |
| 通义千问 DashScope（兼容模式） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Moonshot | `https://api.moonshot.cn/v1` |
| 本地 vLLM / Ollama（OpenAI 模式） | `http://127.0.0.1:8000/v1` |

模型名填对应服务商控制台里的名称即可（例如 `gpt-4o-mini`、`deepseek-chat`、
`qwen-vl-max`）。支持多模态的服务可以把 `include_image: true` 打开，
策略会额外附上一张 96×96 的高度图 PNG。

## 3. 配置（两步）

**第一步：填写密钥**（密钥不写进 YAML）

```bash
cp .env.example .env
nano .env          # 填 AGENT_API_KEY，必要时改 AGENT_API_BASE / AGENT_API_MODEL
```

**第二步：选择实验参数**

```yaml
# config/experiment.yaml
scene:
  objects: 5
policy:
  kind: vlm                 # greedy | checkpoint | vlm
  vlm:
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    api_key_env: "AGENT_API_KEY"
    temperature: 0.0
    max_tokens: 300
    timeout_s: 30
    retries: 2
    include_image: false
logging:
  out_dir: "results/api_runs"
safety:
  fallback: greedy
```

配置优先级：命令行 > 环境变量 > `experiment.yaml` > 内置默认值。

## 4. 运行

```bash
# 先验证 key / endpoint / model（一次极小的请求）
source .env
bash scripts/check_api.sh

# 运行 API 策略实验（使用 config/experiment.yaml，并强制 policy=vlm）
bash scripts/run_vlm_demo.sh
```

也可以手动指定：

```bash
ros2 launch ag_agent demo.launch.py policy:=vlm \
  config:=$HOME/ros2_ws/src/active-grasp/config/experiment.yaml
```

## 5. 结果在哪里

```
results/api_runs/demo_vlm_<timestamp>.jsonl
```

每行是一条决策记录：

```json
{
  "policy": "vlm",
  "step": 3,
  "action": "PUSH cell (30, 22) dir 6",
  "reward": 0.06,
  "cleared": 1,
  "objects": 5,
  "api": {"calls": 3, "errors": 0, "fallbacks": 0, "mean_latency_s": 1.8},
  "summary": {"peaks": [], "mean_uncertainty_in_window": 0.31}
}
```

终端还会打印：

```
[demo] api stats: {'calls': 25, 'errors': 0, 'fallbacks': 0, 'mean_latency_s': 1.9}
```

## 6. 怎么把它做成"可写简历"的对照实验

1. **同场景、同种子**：`scene.objects` 与随机种子保持一致；
2. 至少 3 个策略：`greedy`、`checkpoint`（DDQN）、`vlm`；
3. 每个条件跑 **5 个随机种子 × 10 个 episode**，记录：
   - 清理率 `cleared / objects`
   - 平均步数
   - 平均 API 延迟、失败率、回退率
4. 报告均值 ± 标准差，并说明 VLM 在**高杂乱度**（8 物体）时是否更有优势；
5. 若开了 `view` 动作，统计"看"的比例与信息增益（不确定度下降）。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `API FAILED: 401` | key 错误或没有 `source .env` |
| `404 Not Found` | `base_url` 少了 `/v1` 或服务商路径不同 |
| `404 model not found` | `model` 名字写错（用控制台里的完整名称） |
| 大量 `fallbacks` | 看 `API policy failed (...)` 的原因；多为 JSON 不合法，可把 `max_tokens` 调大 |
| 延迟很高 | 换更小/更快的模型，或把 `include_image` 关掉 |
| 费用担心 | 先用 `scene.objects: 2`、`max_steps: 8` 跑通，再扩大规模 |

## 8. 安全与复现

- `.env` 已被 `.gitignore` 忽略，**不要**把 key 提交到 GitHub；
- 每次实验的配置会以脱敏形式打印到日志（key 显示为 `***`）；
- JSONL 可直接用于后续画图/统计（例如 `pandas.read_json(..., lines=True)`）。
