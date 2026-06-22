import argparse
import csv

import numpy as np

from config import Config
from environment import AirGroundMECEnv
from ippo import IPPO
from utils import ensure_dirs, resolve_device, set_seed


def evaluate(args):
    cfg = Config()

    cfg.env.seed = args.seed
    cfg.env.n_devices = args.devices or cfg.env.n_devices
    cfg.env.n_uavs = args.uavs or cfg.env.n_uavs

    set_seed(args.seed)

    ensure_dirs(cfg.paths.results)

    env = AirGroundMECEnv(cfg.env, cfg.reward)

    device = resolve_device(cfg.train.device)

    algo = IPPO(
        env.obs_dim,
        env.global_state_dim,
        env.action_dim,
        cfg.train,
        device,
    )

    algo.load(args.checkpoint)

    rows = []

    for episode in range(1, args.episodes + 1):

        obs, _ = env.reset(seed=args.seed + 10000 + episode)

        ep_reward = 0.0

        metrics = {
            "queue_delay": [],
            "energy": [],
            "fairness": [],
            "resource_utilization": [],
            "throughput": [],
        }

        done = False

        while not done:

            action, _, _ = algo.act(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            ep_reward += reward

            for key in metrics:
                metrics[key].append(info[key])

        rows.append(
            {
                "episode": episode,
                "reward": ep_reward,
                **{
                    k: float(np.mean(v))
                    for k, v in metrics.items()
                },
            }
        )

    output = cfg.paths.results / "ippo_evaluation_metrics.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluation over {args.episodes} episodes")

    for key in rows[0]:
        if key != "episode":
            print(
                f"{key}: "
                f"{np.mean([r[key] for r in rows]):.4f}"
            )

    print(f"Saved metrics: {output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ippo_final.pt"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20
    )

    parser.add_argument(
        "--devices",
        type=int,
        default=None
    )

    parser.add_argument(
        "--uavs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17
    )

    evaluate(parser.parse_args())