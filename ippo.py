from dataclasses import dataclass

import numpy as np
import torch
from agents import SharedAgentActor, MLP
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig

class IndependentCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.value = MLP(obs_dim, hidden_dim, 1)

    def forward(self, obs):
        return self.value(obs).squeeze(-1)

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, state, action, log_prob, reward, done, value):
        self.obs.append(obs.copy())
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.log_probs.append(log_prob.copy())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def clear(self):
        self.__init__()

    def compute_returns_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return returns, advantages

    def as_batch(self, device, gamma: float, gae_lambda: float, last_value: float):
        returns, advantages = self.compute_returns_advantages(last_value, gamma, gae_lambda)
        obs = torch.tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        states = torch.tensor(np.asarray(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.asarray(self.actions), dtype=torch.float32, device=device)
        log_probs = torch.tensor(np.asarray(self.log_probs), dtype=torch.float32, device=device)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return RolloutBatch(obs, states, actions, log_probs, returns, advantages)


class IPPO:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: TrainConfig, device):
        self.cfg = cfg
        self.device = device
        self.actor = SharedAgentActor(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic = IndependentCritic(obs_dim, cfg.hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate,
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs, _ = self.actor(obs_t)
        value = self.critic(obs_t).mean().item()
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            value,
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return float(self.critic(obs_t).mean().item())
    
    def update(self, batch: RolloutBatch):
        t, n, obs_dim = batch.obs.shape
        action_dim = batch.actions.shape[-1]

        flat_obs = batch.obs.reshape(t * n, obs_dim)
        flat_actions = batch.actions.reshape(t * n, action_dim)
        flat_old_log_probs = batch.log_probs.reshape(t * n)
        flat_advantages = batch.advantages[:, None].repeat(1, n).reshape(t * n)
        flat_returns = batch.returns[:, None].repeat(1, n).reshape(t * n)

        dataset = TensorDataset(
            flat_obs,
            flat_actions,
            flat_old_log_probs,
            flat_advantages,
            flat_returns,
        )

        loader = DataLoader(dataset, batch_size=min(self.cfg.minibatch_size, len(dataset)), shuffle=True)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for obs, actions, old_log_probs, advantages, returns in loader:
                _, new_log_probs, entropy = self.actor(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                values = self.critic(obs)
                value_loss = F.mse_loss(values, returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_loss.item())
                updates += 1

        return {k: v / max(updates, 1) for k, v in metrics.items()}

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
