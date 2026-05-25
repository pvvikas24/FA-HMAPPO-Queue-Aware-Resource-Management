import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class SharedAgentActor(nn.Module):
    """Parameter-shared decentralized actor for all IoT agents."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)

    def distribution(self, obs):
        h = torch.tanh(self.backbone(obs))
        alpha = F.softplus(self.alpha_head(h)) + 1.01
        beta = F.softplus(self.beta_head(h)) + 1.01
        return Beta(alpha, beta)

    def forward(self, obs, action=None):
        dist = self.distribution(obs)
        if action is None:
            action = dist.rsample()
        action = torch.clamp(action, 1e-5, 1.0 - 1e-5)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class CentralizedCritic(nn.Module):
    """Centralized value function over the concatenated multi-agent state."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.value = MLP(state_dim, hidden_dim, 1)

    def forward(self, state):
        return self.value(state).squeeze(-1)

