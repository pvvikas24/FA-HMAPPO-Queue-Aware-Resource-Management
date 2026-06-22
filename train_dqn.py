import argparse
import csv

import numpy as np

from config import Config
from environment import AirGroundMECEnv
from dqn import DQN
from plots import plot_training_curves
from utils import ensure_dirs, resolve_device, save_json, set_seed


def train(args):
    cfg = Config()
    cfg.train.episodes = args.episodes or cfg.train.episodes
    cfg.env.n_devices = args.devices or cfg.env.n_devices
    cfg.env.n_uavs = args.uavs or cfg.env.n_uavs
    cfg.env.seed = args.seed

    set_seed(args.seed)

    ensure_dirs(
        cfg.paths.checkpoints,
        cfg.paths.results,
        cfg.paths.figures,
    )

    env = AirGroundMECEnv(cfg.env, cfg.reward)

    device = resolve_device(cfg.train.device)

    algo = DQN(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.train.hidden_dim,
        device=device,
    )

    history = []

    for episode in range(1, cfg.train.episodes + 1):

        obs, _ = env.reset(seed=args.seed + episode)

        ep = {
            "reward": 0.0,
            "queue_delay": [],
            "energy": [],
            "fairness": [],
            "resource_utilization": [],
            "throughput": [],
        }

        done = False

        for _ in range(cfg.train.rollout_steps):

            action, action_idx = algo.select_actions(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            algo.store_transition(
                obs,
                action_idx,
                reward,
                next_obs,
                done,
            )

            obs = next_obs

            ep["reward"] += reward

            for key in ep:
                if key != "reward":
                    ep[key].append(info[key])

            if done:
                break

        loss = algo.train_step()

        losses = {
            "policy_loss": loss,
            "value_loss": 0.0,
            "entropy": 0.0,
        }

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

        if episode % 10 == 0:
            algo.update_target()

        if episode % args.log_every == 0:
            print(
                f"Episode {episode:04d} | "
                f"reward={row['reward']:.3f} "
                f"delay={row['queue_delay']:.3f} "
                f"fairness={row['fairness']:.3f} "
                f"throughput={row['throughput']:.3f}"
            )

        if episode % cfg.train.checkpoint_every == 0:
            algo.save(
                cfg.paths.checkpoints / f"dqn_ep{episode}.pt"
            )

    final_path = cfg.paths.checkpoints / "dqn_final.pt"

    algo.save(final_path)

    metrics_path = cfg.paths.results / "dqn_training_metrics.csv"

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys())
        )

        writer.writeheader()
        writer.writerows(history)

    save_json(
        cfg.paths.results / "run_config.json",
        {
            "episodes": cfg.train.episodes,
            "devices": cfg.env.n_devices,
            "uavs": cfg.env.n_uavs,
        },
    )

    plot_training_curves(
        metrics_path,
        cfg.paths.figures,
    )

    print(f"Saved checkpoint: {final_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--uavs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )

    train(parser.parse_args())