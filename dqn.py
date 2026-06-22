from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_MAP = np.array([
    [0.00, 0.00, 0.50, 0.50],
    [0.25, 0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50, 0.50],
    [0.75, 0.00, 0.50, 0.50],
    [1.00, 0.00, 0.50, 0.50],

    [0.50, 0.50, 0.50, 0.50],
    [1.00, 0.50, 0.50, 0.50],

    [0.50, 1.00, 0.50, 0.50],
    [1.00, 1.00, 0.50, 0.50],
], dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQN:
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        device,
        lr=1e-4,
        gamma=0.99,
    ):
        self.device = device
        self.gamma = gamma

        self.n_actions = len(ACTION_MAP)

        self.q_net = QNetwork(
            obs_dim,
            hidden_dim,
            self.n_actions
        ).to(device)

        self.target_net = QNetwork(
            obs_dim,
            hidden_dim,
            self.n_actions
        ).to(device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=lr
        )

        self.replay_buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_actions(self, obs):
        n_agents = obs.shape[0]

        actions = np.zeros((n_agents, 4), dtype=np.float32)
        action_indices = np.zeros(n_agents, dtype=np.int64)

        for i in range(n_agents):

            if random.random() < self.epsilon:
                idx = random.randint(
                    0,
                    self.n_actions - 1
                )
            else:
                state = torch.tensor(
                    obs[i],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    q_values = self.q_net(state)

                idx = int(torch.argmax(q_values).item())

            action_indices[i] = idx
            actions[i] = ACTION_MAP[idx]

        return actions, action_indices

    def store_transition(
        self,
        obs,
        action_idx,
        reward,
        next_obs,
        done
    ):
        n_agents = obs.shape[0]

        for i in range(n_agents):
            self.replay_buffer.push(
                obs[i],
                action_idx[i],
                reward,
                next_obs[i],
                done
            )

    def train_step(self, batch_size=256):

        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(batch_size)
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        q_values = self.q_net(states)
        current_q = q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target_net(
                next_states
            ).max(1)[0]

            target_q = rewards + (
                1.0 - dones
            ) * self.gamma * next_q

        loss = F.mse_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

    def save(self, path):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.q_net.load_state_dict(
            checkpoint["q_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )