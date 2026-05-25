from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from utils import moving_average


def _plot(df, x, y, output_dir: Path, ylabel: str):
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], alpha=0.35, label="raw")
    ma = moving_average(df[y].to_numpy(), window=10)
    if len(ma):
        plt.plot(df[x].iloc[-len(ma) :], ma, linewidth=2, label="moving avg")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{y}.png", dpi=180)
    plt.close()


def plot_training_curves(metrics_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    labels = {
        "reward": "Average Reward",
        "queue_delay": "Average Queue Delay",
        "energy": "Energy Consumption",
        "fairness": "Jain Fairness Index",
        "throughput": "Throughput",
        "resource_utilization": "Resource Utilization",
    }
    for column, label in labels.items():
        _plot(df, "episode", column, output_dir, label)


if __name__ == "__main__":
    plot_training_curves("results/training_metrics.csv", "figures")
