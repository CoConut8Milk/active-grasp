"""Realtime demo: greedy baseline, trained DDQN or an API-backed policy.

The policy is selected by ``--policy`` or ``policy.kind`` in
``config/experiment.yaml``. Every step is written to a JSONL log so that API
experiments can be analysed afterwards.
"""

import argparse
import json
import pathlib
import time

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from ag_agent.agent import DQNAgent
from ag_agent.baselines import GreedyPolicy
from ag_agent.config import load_config, redacted
from ag_agent.env import ActiveGraspEnv
from ag_agent.vlm_policy import VLMPolicy


def describe(action_space, action):
    kind, *payload = action_space.decode(action)
    if kind == "grasp":
        return f"GRASP cell ({payload[0]}, {payload[1]})"
    if kind == "push":
        return f"PUSH cell ({payload[0]}, {payload[1]}) dir {payload[2]}"
    return f"VIEW {payload[0]}"


def build_policy(kind, env, checkpoint, config, logger):
    if kind == "checkpoint":
        if not checkpoint:
            raise SystemExit("--policy checkpoint requires --checkpoint <file>")
        agent = DQNAgent(env.action_space)
        agent.load(checkpoint)
        return agent, "checkpoint"
    if kind == "vlm":
        return VLMPolicy(env.action_space, config["policy"]["vlm"], logger=logger), "vlm"
    return GreedyPolicy(env.action_space), "greedy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="path to config/experiment.yaml")
    parser.add_argument("--policy", default="",
                        choices=["", "greedy", "checkpoint", "vlm"],
                        help="overrides policy.kind from the config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--objects", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--step-delay", type=float, default=0.5)
    parser.add_argument("--out", default=None, help="JSONL output directory")
    args = parser.parse_args(remove_ros_args()[1:])

    config = load_config(args.config)
    policy_kind = args.policy or config["policy"]["kind"]
    checkpoint = args.checkpoint or config["policy"].get("checkpoint") or None
    objects = args.objects if args.objects is not None else int(config["scene"]["objects"])

    rclpy.init()
    node = Node("ag_demo")
    node.get_logger().info(f"policy={policy_kind} objects={objects}")
    node.get_logger().info(f"config={json.dumps(redacted(config), ensure_ascii=False)}")

    env = ActiveGraspEnv(node, height=48, width=48, n_dirs=8, n_views=4)
    policy, policy_kind = build_policy(
        policy_kind, env, checkpoint, config, node.get_logger()
    )

    out_dir = pathlib.Path(args.out or config["logging"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"demo_{policy_kind}_{int(time.time())}.jsonl"
    save_jsonl = bool(config["logging"].get("save_jsonl", True))
    log_handle = open(log_path, "w", encoding="utf-8") if save_jsonl else None

    state = env.reset(objects)
    done = False
    try:
        while not done:
            mask = env.valid_mask()
            if hasattr(policy, "select"):
                action = policy.select(
                    env.state_tensor(), mask, args.epsilon, eval_mode=True
                )
            else:
                action = policy.act(state, mask)
            print(f"[demo] {describe(env.action_space, action)}")
            next_state, reward, done, info = env.step(action)
            if hasattr(policy, "observe_result"):
                policy.observe_result(action, info.get("success", False))
            print(
                f"[demo] reward={reward:.2f} cleared={info['cleared']}/{env.n_objects}"
            )
            if log_handle is not None:
                record = {
                    "policy": policy_kind,
                    "step": int(info.get("steps", 0)),
                    "action": describe(env.action_space, action),
                    "reward": float(reward),
                    "cleared": int(info.get("cleared", 0)),
                    "objects": int(env.n_objects),
                    "info": {
                        key: value
                        for key, value in info.items()
                        if isinstance(value, (int, float, bool))
                    },
                }
                if isinstance(policy, VLMPolicy):
                    record["api"] = policy.summary_stats()
                    record["summary"] = policy.last_summary
                log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_handle.flush()
            state = next_state
            time.sleep(args.step_delay)
    finally:
        print(f"[demo] episode done: cleared {env.cleared}/{env.n_objects}")
        if isinstance(policy, VLMPolicy):
            print(f"[demo] api stats: {policy.summary_stats()}")
        if log_handle is not None:
            log_handle.close()
            node.get_logger().info(f"wrote {log_path}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
