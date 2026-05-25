from dataclasses import dataclass

import numpy as np

from config import RewardConfig
from fairness import delay_fairness, throughput_fairness


@dataclass
class RewardBreakdown:
    reward: float
    queue_delay: float
    energy: float
    fairness: float
    utilization: float
    throughput: float
    hotspot_imbalance: float


def compute_reward(metrics: dict, cfg: RewardConfig) -> RewardBreakdown:
    delays = np.asarray(metrics["device_delays"], dtype=np.float64)
    throughputs = np.asarray(metrics["device_throughputs"], dtype=np.float64)
    energy = float(metrics["energy"])
    utilization = float(metrics["resource_utilization"])
    throughput = float(metrics["throughput"])
    hotspot_imbalance = float(metrics["hotspot_imbalance"])

    queue_delay = float(np.mean(delays)) if delays.size else 0.0
    fairness = 0.5 * delay_fairness(delays) + 0.5 * throughput_fairness(throughputs)

    reward = (
        -cfg.alpha_delay * queue_delay
        -cfg.beta_energy * energy
        +cfg.gamma_fairness * fairness
        +cfg.delta_utilization * utilization
        +cfg.eta_throughput * throughput
        -cfg.hotspot_penalty * hotspot_imbalance
    )
    return RewardBreakdown(
        reward=float(reward),
        queue_delay=queue_delay,
        energy=energy,
        fairness=float(fairness),
        utilization=utilization,
        throughput=throughput,
        hotspot_imbalance=hotspot_imbalance,
    )

