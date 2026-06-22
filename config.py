from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    n_devices: int = 12
    n_uavs: int = 5
    area_size_m: float = 1000.0
    episode_length: int = 120
    slot_time_s: float = 1.0
    seed: int = 7

    task_arrival_prob: float = 0.65
    min_task_size_bits: float = 0.5e6
    max_task_size_bits: float = 4.0e6
    min_cycles_per_bit: float = 500.0
    max_cycles_per_bit: float = 1400.0
    max_queue_bits: float = 80e6

    device_cpu_hz: float = 0.8e9
    uav_cpu_hz: float = 8.0e9
    haps_cpu_hz: float = 28.0e9
    bandwidth_hz: float = 8.0e6
    tx_power_device_w: float = 0.4
    tx_power_uav_w: float = 1.5
    noise_power_w: float = 1e-12
    path_loss_exp: float = 2.2
    uav_altitude_m: float = 120.0
    haps_altitude_m: float = 20_000.0

    uav_speed_mps: float = 18.0
    uav_mobility_alpha: float = 0.35
    battery_j: float = 25_000.0
    haps_backhaul_bandwidth_hz: float = 25.0e6


@dataclass
class RewardConfig:
    alpha_delay: float = 1.0
    beta_energy: float = 0.25
    gamma_fairness: float = 1.5
    delta_utilization: float = 0.8
    eta_throughput: float = 0.5
    hotspot_penalty: float = 0.35


@dataclass
class TrainConfig:
    episodes: int = 250
    rollout_steps: int = 120
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 6
    minibatch_size: int = 512
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    checkpoint_every: int = 25
    device: str = "auto"


@dataclass
class PathsConfig:
    root: Path = Path(__file__).resolve().parent
    checkpoints: Path = root / "checkpoints"
    results: Path = root / "results"
    figures: Path = root / "figures"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
