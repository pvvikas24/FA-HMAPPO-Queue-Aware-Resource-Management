import argparse
import csv

import numpy as np

from config import Config
from environment import AirGroundMECEnv
from mappo import MAPPO, RolloutBuffer
from plots import plot_training_curves
from utils import ensure_dirs, resolve_device, save_json, set_seed


def train(args):
    cfg = Config()
    cfg.train.episodes = args.episodes or cfg.train.episodes
    cfg.env.n_devices = args.devices or cfg.env.n_devices
    cfg.env.n_uavs = args.uavs or cfg.env.n_uavs
    cfg.env.seed = args.seed
    set_seed(args.seed)
    ensure_dirs(cfg.paths.checkpoints, cfg.paths.results, cfg.paths.figures)

    env = AirGroundMECEnv(cfg.env, cfg.reward)
    device = resolve_device(cfg.train.device)
    algo = MAPPO(env.obs_dim, env.global_state_dim, env.action_dim, cfg.train, device)
    history = []

    for episode in range(1, cfg.train.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        buffer = RolloutBuffer()
        ep = {
            "reward": 0.0,
            "queue_delay": [],
            "energy": [],
            "fairness": [],
            "resource_utilization": [],
            "throughput": [],
        }

        for _ in range(cfg.train.rollout_steps):
            action, log_prob, value = algo.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, obs.flatten(), action, log_prob, reward, done, value)
            obs = next_obs

            ep["reward"] += reward
            for key in ep:
                if key != "reward":
                    ep[key].append(info[key])
            if done:
                break

        last_value = 0.0 if done else algo.value(obs)
        batch = buffer.as_batch(device, cfg.train.gamma, cfg.train.gae_lambda, last_value)
        losses = algo.update(batch)
        row = {
            "episode": episode,
            "reward": ep["reward"],
            "queue_delay": float(np.mean(ep["queue_delay"])),
            "energy": float(np.mean(ep["energy"])),
            "fairness": float(np.mean(ep["fairness"])),
            "resource_utilization": float(np.mean(ep["resource_utilization"])),
            "throughput": float(np.mean(ep["throughput"])),
            **losses,
        }
        history.append(row)

        if episode % args.log_every == 0:
            print(
                f"Episode {episode:04d} | reward={row['reward']:.3f} "
                f"delay={row['queue_delay']:.3f} fairness={row['fairness']:.3f} "
                f"throughput={row['throughput']:.3f}"
            )
        if episode % cfg.train.checkpoint_every == 0:
            algo.save(cfg.paths.checkpoints / f"fahmappo_ep{episode}.pt")

    final_path = cfg.paths.checkpoints / "fahmappo_final.pt"
    algo.save(final_path)
    metrics_path = cfg.paths.results / "training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_json(
        cfg.paths.results / "run_config.json",
        {"episodes": cfg.train.episodes, "devices": cfg.env.n_devices, "uavs": cfg.env.n_uavs},
    )
    plot_training_curves(metrics_path, cfg.paths.figures)
    print(f"Saved checkpoint: {final_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--uavs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    train(parser.parse_args())

