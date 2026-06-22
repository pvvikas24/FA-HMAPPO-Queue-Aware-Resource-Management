import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

fahmappo = pd.read_csv("results/training_metrics.csv")
ippo = pd.read_csv("results/ippo_training_metrics.csv")
dqn = pd.read_csv("results/dqn_training_metrics.csv")

# Update these if you have actual values
GREEDY = {
    "reward": 80,
    "queue_delay": 0.90,
    "fairness": 0.80,
    "resource_utilization": 0.55,
    "energy": 0.070,
    "throughput": 0.38,
}

RANDOM = {
    "reward": 40,
    "queue_delay": 1.30,
    "fairness": 0.72,
    "resource_utilization": 0.45,
    "energy": 0.090,
    "throughput": 0.30,
}

metrics = {
    "reward": {
        "title": "Reward Convergence Comparison",
        "ylabel": "Reward"
    },
    "queue_delay": {
        "title": "Queue Delay Convergence Comparison",
        "ylabel": "Queue Delay"
    },
    "fairness": {
        "title": "Fairness Convergence Comparison",
        "ylabel": "Fairness"
    },
    "energy": {
        "title": "Energy Consumption Convergence Comparison",
        "ylabel": "Energy"
    },
    "resource_utilization": {
        "title": "Resource Utilization Convergence Comparison",
        "ylabel": "Resource Utilization"
    },
    "throughput": {
        "title": "Throughput Convergence Comparison",
        "ylabel": "Throughput"
    }
}

for metric, cfg in metrics.items():

    plt.figure(figsize=(16, 9))

    plt.plot(
        fahmappo["episode"],
        fahmappo[metric].rolling(10, min_periods=1).mean(),
        linewidth=3,
        label="FA-HMAPPO"
    )

    plt.plot(
        ippo["episode"],
        ippo[metric].rolling(10, min_periods=1).mean(),
        linewidth=3,
        label="IPPO"
    )

    plt.plot(
        dqn["episode"],
        dqn[metric].rolling(10, min_periods=1).mean(),
        linewidth=3,
        label="DQN"
    )

    plt.axhline(
        GREEDY[metric],
        linestyle="--",
        linewidth=3,
        label="Greedy"
    )

    plt.axhline(
        RANDOM[metric],
        linestyle=":",
        linewidth=3,
        label="Random"
    )

    plt.title(
        cfg["title"],
        fontsize=24,
        fontweight="bold"
    )

    plt.xlabel(
        "Training Episodes",
        fontsize=18,
        fontweight="bold"
    )

    plt.ylabel(
        cfg["ylabel"],
        fontsize=18,
        fontweight="bold"
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.legend(
        fontsize=14,
        loc="best"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{metric}_comparison.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

print("\nAll graphs generated successfully!")
print("Saved in output folder:")
print("reward_comparison.png")
print("queue_delay_comparison.png")
print("fairness_comparison.png")
print("energy_comparison.png")
print("resource_utilization_comparison.png")
print("throughput_comparison.png")