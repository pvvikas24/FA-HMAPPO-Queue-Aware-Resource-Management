from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import EnvConfig, RewardConfig
from reward import compute_reward


class AirGroundMECEnv(gym.Env):
    """Queue-aware UAV-assisted MEC environment for FA-HMAPPO experiments.

    Agents are IoT devices. A continuous action vector is decoded as:
    [offload_ratio, target_selector, resource_request, scheduling_priority].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()
        self.cfg = env_config or EnvConfig()
        self.reward_cfg = reward_config or RewardConfig()
        self.n = self.cfg.n_devices
        self.m = self.cfg.n_uavs
        self.obs_dim = 15
        self.action_dim = 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n, self.obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n, self.action_dim), dtype=np.float32
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset(seed=self.cfg.seed)

    @property
    def global_state_dim(self) -> int:
        return self.n * self.obs_dim

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        self.t = 0
        self.device_pos = self.rng.uniform(0, c.area_size_m, size=(self.n, 2))
        self.uav_pos = self.rng.uniform(0.2, 0.8, size=(self.m, 2)) * c.area_size_m
        self.haps_pos = np.array([c.area_size_m / 2, c.area_size_m / 2])
        self.queue_bits = np.zeros(self.n, dtype=np.float64)
        self.queue_cycles = np.zeros(self.n, dtype=np.float64)
        self.last_task_bits = np.zeros(self.n, dtype=np.float64)
        self.last_delay = np.zeros(self.n, dtype=np.float64)
        self.last_throughput = np.zeros(self.n, dtype=np.float64)
        self.energy_level = np.full(self.n, c.battery_j, dtype=np.float64)
        self.uav_cpu_available = np.full(self.m, c.uav_cpu_hz, dtype=np.float64)
        self.haps_cpu_available = c.haps_cpu_hz
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        if action.shape != (self.n, self.action_dim):
            action = action.reshape(self.n, self.action_dim)

        self._generate_tasks()
        self._move_uavs_toward_hotspots()

        offload_ratio = action[:, 0]
        target = np.floor(action[:, 1] * (self.m + 1)).astype(int)
        target = np.clip(target, 0, self.m)
        resource_request = 0.05 + action[:, 2]
        priority = action[:, 3]

        served_bits = np.zeros(self.n, dtype=np.float64)
        delays = np.zeros(self.n, dtype=np.float64)
        energy = np.zeros(self.n, dtype=np.float64)
        cpu_used_uav = np.zeros(self.m, dtype=np.float64)
        cpu_used_haps = 0.0

        cpb = self._cycles_per_bit()
        local_bits = self.queue_bits * (1.0 - offload_ratio)
        local_cycles_requested = local_bits * cpb
        local_cycles_done = np.minimum(local_cycles_requested, c.device_cpu_hz * c.slot_time_s)
        local_served = local_cycles_done / np.maximum(cpb, 1.0)
        local_delay = np.divide(
            local_served,
            np.maximum(c.device_cpu_hz / np.maximum(cpb, 1.0), 1.0),
            out=np.zeros_like(local_served),
            where=local_served > 0,
        )
        local_energy = 1e-27 * (c.device_cpu_hz**2) * local_cycles_done

        served_bits += local_served
        delays += local_delay
        energy += local_energy

        for uav_id in range(self.m):
            members = np.where((target == uav_id) & (offload_ratio > 0.02))[0]
            if members.size:
                done, delay, e, used = self._serve_edge_group(
                    members, uav_id, resource_request, priority, cpb
                )
                served_bits[members] += done
                delays[members] += delay
                energy[members] += e
                cpu_used_uav[uav_id] += used

        haps_members = np.where((target == self.m) & (offload_ratio > 0.02))[0]
        if haps_members.size:
            done, delay, e, used = self._serve_haps_group(
                haps_members, resource_request, priority, cpb
            )
            served_bits[haps_members] += done
            delays[haps_members] += delay
            energy[haps_members] += e
            cpu_used_haps += used

        served_bits = np.minimum(served_bits, self.queue_bits)
        served_cycles = served_bits * cpb
        self.queue_bits = np.maximum(self.queue_bits - served_bits, 0.0)
        self.queue_cycles = np.maximum(self.queue_cycles - served_cycles, 0.0)
        self.energy_level = np.maximum(self.energy_level - energy, 0.0)
        self.last_delay = delays + self.queue_bits / np.maximum(served_bits, 1.0)
        self.last_throughput = served_bits / c.slot_time_s

        queue_load_by_uav = self._uav_queue_loads(target)
        hotspot_imbalance = float(np.std(queue_load_by_uav) / (np.mean(queue_load_by_uav) + 1e-8))
        total_cpu = self.m * c.uav_cpu_hz + c.haps_cpu_hz + self.n * c.device_cpu_hz
        used_cpu = np.sum(local_cycles_done) / c.slot_time_s + np.sum(cpu_used_uav) + cpu_used_haps
        utilization = float(np.clip(used_cpu / total_cpu, 0.0, 1.0))
        throughput_norm = float(np.sum(served_bits) / (self.n * c.max_task_size_bits))
        energy_norm = float(np.mean(energy) / 10.0)
        delay_norm = self.last_delay / 10.0

        reward_breakdown = compute_reward(
            {
                "device_delays": delay_norm,
                "device_throughputs": self.last_throughput / c.max_task_size_bits,
                "energy": energy_norm,
                "resource_utilization": utilization,
                "throughput": throughput_norm,
                "hotspot_imbalance": hotspot_imbalance,
            },
            self.reward_cfg,
        )

        self.t += 1
        terminated = False
        truncated = self.t >= c.episode_length
        obs = self._get_obs()
        info = {
            "queue_delay": reward_breakdown.queue_delay,
            "energy": reward_breakdown.energy,
            "fairness": reward_breakdown.fairness,
            "resource_utilization": reward_breakdown.utilization,
            "throughput": reward_breakdown.throughput,
            "average_reward": reward_breakdown.reward,
            "raw_queue_bits": float(np.mean(self.queue_bits)),
            "hotspot_imbalance": hotspot_imbalance,
        }
        return obs, reward_breakdown.reward, terminated, truncated, info

    def _generate_tasks(self) -> None:
        c = self.cfg
        arrivals = self.rng.random(self.n) < c.task_arrival_prob
        sizes = self.rng.uniform(c.min_task_size_bits, c.max_task_size_bits, self.n) * arrivals
        cpb = self.rng.uniform(c.min_cycles_per_bit, c.max_cycles_per_bit, self.n)
        self.queue_bits = np.minimum(self.queue_bits + sizes, c.max_queue_bits)
        self.queue_cycles = np.minimum(
            self.queue_cycles + sizes * cpb, c.max_queue_bits * c.max_cycles_per_bit
        )
        self.last_task_bits = sizes

    def _cycles_per_bit(self):
        return np.divide(
            self.queue_cycles,
            np.maximum(self.queue_bits, 1.0),
            out=np.full(self.n, self.cfg.min_cycles_per_bit, dtype=np.float64),
            where=self.queue_bits > 1.0,
        )

    def _serve_edge_group(self, members, uav_id, request, priority, cpb):
        c = self.cfg
        weights = request[members] * (0.2 + priority[members])
        weights = weights / (np.sum(weights) + 1e-8)
        cpu_alloc = weights * c.uav_cpu_hz
        candidate_bits = self.queue_bits[members] * np.clip(request[members], 0.0, 1.0)
        tx_rate = self._uplink_rate_to_uav(members, uav_id)
        tx_limited = tx_rate * c.slot_time_s
        cpu_limited = cpu_alloc * c.slot_time_s / np.maximum(cpb[members], 1.0)
        done = np.minimum.reduce([candidate_bits, tx_limited, cpu_limited])
        tx_delay = done / np.maximum(tx_rate, 1.0)
        comp_delay = done * cpb[members] / np.maximum(cpu_alloc, 1.0)
        energy = c.tx_power_device_w * tx_delay
        cpu_used = float(np.sum(done * cpb[members] / c.slot_time_s))
        return done, tx_delay + comp_delay, energy, cpu_used

    def _serve_haps_group(self, members, request, priority, cpb):
        c = self.cfg
        relay = self._nearest_uav(members)
        weights = request[members] * (0.2 + priority[members])
        weights = weights / (np.sum(weights) + 1e-8)
        cpu_alloc = weights * c.haps_cpu_hz
        candidate_bits = self.queue_bits[members] * np.clip(request[members], 0.0, 1.0)
        access_rate = np.array([self._uplink_rate_to_uav([i], relay[j])[0] for j, i in enumerate(members)])
        backhaul_rate = c.haps_backhaul_bandwidth_hz / max(len(members), 1)
        cpu_limited = cpu_alloc * c.slot_time_s / np.maximum(cpb[members], 1.0)
        backhaul_limited = np.full_like(candidate_bits, backhaul_rate * c.slot_time_s)
        done = np.minimum.reduce(
            [candidate_bits, access_rate * c.slot_time_s, backhaul_limited, cpu_limited]
        )
        tx_delay = done / np.maximum(access_rate, 1.0) + done / max(backhaul_rate, 1.0)
        comp_delay = done * cpb[members] / np.maximum(cpu_alloc, 1.0)
        energy = c.tx_power_device_w * (done / np.maximum(access_rate, 1.0)) + c.tx_power_uav_w * (done / max(backhaul_rate, 1.0))
        cpu_used = float(np.sum(done * cpb[members] / c.slot_time_s))
        return done, tx_delay + comp_delay, energy, cpu_used

    def _uplink_rate_to_uav(self, device_indices, uav_id: int):
        c = self.cfg
        idx = np.asarray(device_indices, dtype=int)
        horizontal = np.linalg.norm(self.device_pos[idx] - self.uav_pos[uav_id], axis=1)
        distance = np.sqrt(horizontal**2 + c.uav_altitude_m**2)
        gain = 1.0 / np.maximum(distance, 1.0) ** c.path_loss_exp
        snr = c.tx_power_device_w * gain / c.noise_power_w
        shared_bw = c.bandwidth_hz / max(len(idx), 1)
        return shared_bw * np.log2(1.0 + snr)

    def _nearest_uav(self, device_indices):
        idx = np.asarray(device_indices, dtype=int)
        dist = np.linalg.norm(self.device_pos[idx, None, :] - self.uav_pos[None, :, :], axis=2)
        return np.argmin(dist, axis=1)

    def _uav_queue_loads(self, target):
        loads = np.zeros(self.m, dtype=np.float64)
        nearest = self._nearest_uav(np.arange(self.n))
        for i in range(self.n):
            u = target[i] if target[i] < self.m else nearest[i]
            loads[u] += self.queue_bits[i]
        return loads

    def _move_uavs_toward_hotspots(self) -> None:
        c = self.cfg
        nearest = self._nearest_uav(np.arange(self.n))
        for u in range(self.m):
            members = np.where(nearest == u)[0]
            if members.size == 0:
                target = self.rng.uniform(0, c.area_size_m, size=2)
            else:
                weights = self.queue_bits[members] + 1.0
                target = np.average(self.device_pos[members], axis=0, weights=weights)
            direction = target - self.uav_pos[u]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                step = min(c.uav_speed_mps * c.slot_time_s, norm)
                self.uav_pos[u] += c.uav_mobility_alpha * step * direction / norm
        self.uav_pos = np.clip(self.uav_pos, 0.0, c.area_size_m)

    def _get_obs(self):
        c = self.cfg
        nearest = self._nearest_uav(np.arange(self.n))
        dist = np.linalg.norm(self.device_pos - self.uav_pos[nearest], axis=1)
        uav_loads = self._uav_queue_loads(nearest)
        nearest_load = uav_loads[nearest] / max(c.max_queue_bits * self.n / self.m, 1.0)
        obs = np.column_stack(
            [
                self.queue_bits / c.max_queue_bits,
                self.last_task_bits / c.max_task_size_bits,
                np.full(self.n, c.device_cpu_hz / c.uav_cpu_hz),
                np.full(self.n, c.uav_cpu_hz / c.haps_cpu_hz),
                self.device_pos[:, 0] / c.area_size_m,
                self.device_pos[:, 1] / c.area_size_m,
                self.uav_pos[nearest, 0] / c.area_size_m,
                self.uav_pos[nearest, 1] / c.area_size_m,
                dist / (np.sqrt(2) * c.area_size_m),
                np.full(self.n, c.bandwidth_hz / 1e7),
                self.energy_level / c.battery_j,
                np.clip(self.last_delay / 10.0, 0.0, 1.0),
                np.clip(self.last_throughput / c.max_task_size_bits, 0.0, 1.0),
                np.clip(nearest_load, 0.0, 1.0),
                np.full(self.n, self.t / max(c.episode_length, 1)),
            ]
        )
        return np.clip(obs, 0.0, 1.0).astype(np.float32)
