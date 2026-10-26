# Active-Grasp 交接文档（给下一位助手 / ChatGPT）

> 用途：本文件是一份**自包含的项目上下文**。把本文件（或由
> `scripts/export_context.sh` 生成的压缩包）交给 ChatGPT，它就能在不读取
> 本地 Codex 对话的情况下继续工作。

## 0. 一句话现状

ROS2 Humble + Gazebo Classic 的桌面杂物「抓 / 推 / 换视角」仿真系统已经能
完整启动、感知、规划与控制；**最新一版（commit `79711a7` 之后）修改了相机
朝向、home 位姿、IK 选解与可达窗口，尚未在实验室机器上做一次完整验证**。

## 1. 环境

| 项 | 值 |
|---|---|
| 系统 | Windows 11 + WSL2，Ubuntu 22.04 |
| ROS | Humble（`/opt/ros/humble`） |
| 仿真 | Gazebo Classic 11（WSLg） |
| 用户 / 主机 | `lab_1@DESKTOP-KICTUMK` |
| 项目路径 | `~/ros2_ws/src/active-grasp` |
| 远程仓库 | https://github.com/CoConut8Milk/Active-Grasp |
| 最新提交 | `79711a7`（以 `git log --oneline -1` 为准） |

所有 ROS 命令必须在 **Ubuntu 终端**里执行（提示符 `lab_1@…$`），
不是 Windows 的 `PS C:\…>`。

## 2. 运行方式

```bash
cd ~/ros2_ws/src/active-grasp
git pull
bash scripts/build.sh            # colcon 编译 7 个包

pkill -f gzserver; pkill -f gzclient   # 清理上次残留（否则新 gzserver 会 exit 255）
bash scripts/run_demo.sh         # Gazebo + 感知 + 执行 + 贪心/配置策略
```

启动正常时应依次看到：

1. `Spawn status: SpawnEntity: Successfully spawned entity [ag_arm]`
2. `Configured and activated joint_state_broadcaster / arm_controller / gripper_controller`
3. `execution ready`
4. `perception ready (camera calibrated, signs=(-1.0, +1.0))`
5. `[demo] GRASP cell (u, v)` → `[demo] cleared=x/5`

## 3. 包结构

| 包 | 职责 |
|---|---|
| `ag_interfaces` | Grasp / Push / MoveView / Observe / Reset 服务 |
| `ag_description` | URDF（6-DoF 臂 + 平行夹爪 + 腕部 RGB-D）+ 控制器 YAML |
| `ag_gazebo` | 桌面世界、随机物体生成、spawn 参数 |
| `ag_perception` | 深度反投影、48×48 高度图、三因子不确定度、相机标定 |
| `ag_execution` | URDF 解析、FK/数值 IK、抓/推/看原语、TF 自检 |
| `ag_agent` | DDQN、贪心基线、环境封装、demo、**API 策略（新增）** |
| `ag_bringup` | 顶层 launch |

仓库根目录另有：

- `config/experiment.yaml`：实验唯一入口（场景 / 策略 / API / 日志）
- `scripts/`：`build / run_demo / run_vlm_demo / check_api / train / evaluate`
- `tests/`：35 个纯 Python 单元测试（可在 Windows 本机跑）
- `docs/`：01–09 中文教程（09 是 API 实验手册）

## 4. 关键设计决策（改代码前请先读）

1. **相机朝 -tool z**（与手指同向）。之前相机朝 +tool z，看桌面时夹爪朝天，
   每次抓取都要翻腕 180°。
2. **home 位姿**：相机在 `(0.50, 0, 0.70)` 俯视 `(0.50, 0, 0.02)`；IK 解
   `joint1≈0`，不再整机转身。
3. **关节限位**：joint2 ±2.2、joint3 [-2.2, 1.6]、joint4/5 ±2.6、
   joint1/6 ±π；`max_joint_speed=0.6 rad/s`，大构型变化自动拉长轨迹时长。
4. **IK 选解**：`kinematics.ik_retry` 收集所有收敛解，取「离当前构型最近」
   的那个，并在代价里加 `posture_weight=0.03`，避免肘上/肘下跳变。
5. **夹爪**：掌部 `gripper_base`（11×6×2 cm）+ 两根橙色手指
   （1.6×3.2×7.5 cm），全开 6.4 cm、全闭 0.4 cm，指尖在 tool0 下方 10.5 cm。
6. **抓取下降**：指尖降到桌面上方 1.5 cm（`descend=0.120`），不依赖高度图 z。
7. **接触判定**：先开爪并等宽度稳定，再分 4 步闭合；按「实际宽度变化 <
   期望的 25%」判定被物体挡住。
8. **感知**：`z_max=0.12`（防手臂被当物体）、丢弃深度图最外 3 像素、
   `u_sign=-1 / v_sign=+1`（相机朝 -tool z 后）。
9. **可达窗口**：物体只在 x 0.40–0.62、y ±0.12 内生成；动作掩码限制在
   `u 19–39, v 12–34`。
10. **执行节点线程模型**：`MultiThreadedExecutor(3)` + 服务独立回调组，
    `startup()` 在 executor 启动后执行；不要改回在服务回调里嵌套 spin。

## 5. 最近一次实验（`type (12).txt`）结论

- 启动、控制器、感知、抓取动作全部能跑；
- 日志里每次抓取都有 `stretching move … 3.1 rad`，且偶发
  `actual (-0.51, …) error=1010 mm`：**home 的 joint1=π + 相机朝向相反**
  导致整机转身和翻腕；
- 抓取点有时被推到边界（`grasp cell (44,3)`），超出关节限位后的可达范围。

针对以上问题已提交 `1f56245`（相机同向 + 新 home）、`79711a7`（IK 最近解 +
可达窗口 + 边框夹取），**但还没有拿到这版的新日志**。

## 6. 下一步优先事项

1. 在实验室机器上跑一次最新版（见第 2 节），重点看：
   - `stretching move` 是否 ≤ 1.1 rad；
   - `tool pose … error` 是否 < 20 mm；
   - `jaw contact … width≈0.03` 与 `held=True`；
   - `cleared` 是否开始增加。
2. 若仍夹空：看 `grasp failed: heightmap delta`（大=被推走，≈0=不在指间），
   再决定「两段式接近」或检查 Gazebo 碰撞体。
3. 抓取稳定后：跑贪心 / DDQN / API 三种策略的对照实验（每条件 5 种子 × 10
   episode），记录清理率、步数、API 延迟与回退率，再填简历数字。

## 7. 新 API（DeepSeek / 任意 OpenAI 兼容端点）

```bash
cp .env.example .env      # 填 AGENT_API_KEY
source .env
bash scripts/check_api.sh # 一条极小请求验证 key / base_url / model
bash scripts/run_vlm_demo.sh
```

- 配置在 `config/experiment.yaml` 的 `policy.vlm`；DeepSeek 用
  `https://api.deepseek.com/v1`，模型名以官方控制台为准（代码不写死）。
- 任何 API 失败都会自动回退贪心策略，不会中断仿真。
- 结果写入 `results/api_runs/demo_vlm_<timestamp>.jsonl`。
- **不要把 `.env` 或 key 提交/上传**；`.env` 已在 `.gitignore` 中。

## 8. 已知坑（按症状查）

| 症状 | 处理 |
|---|---|
| `gzserver … exit code 255`、`Controller already loaded` | 上次仿真没退干净：`pkill -f gzserver; pkill -f gzclient` |
| `spawn_entity … Service unavailable` | gzserver 启动慢；launch 已把 `-timeout` 调到 180 s |
| `perception never produced a state` | 先确认 `/ag_camera/depth/image_raw` 有数据、机械臂已 spawn |
| `Position Error … Aborted` | 轨迹过快或 IK 换了构型；检查 `max_joint_speed` 与 IK 最近解 |
| `held=False, width=0.004` | 两指完全闭合但没夹到东西：看 `heightmap delta` 与 `tool pose error` |

## 9. 给 ChatGPT 的开场提示（可直接粘贴）

```text
你接手一个 ROS2 Humble + Gazebo Classic 的机器人抓取项目（Active-Grasp）。
请先阅读附件中的 HANDOFF.md、README.md、config/experiment.yaml 和
docs/09_API策略实验.md，然后：
1) 总结当前系统状态与已知问题；
2) 根据 HANDOFF.md 第 6 节给出下一步验证计划；
3) 不要假设项目已经抓取成功——以最新日志为准。
我接下来会贴出新的运行日志，请基于日志分析，不要凭空猜测。
```
