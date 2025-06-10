import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from agents import BaseAgent

class SpatialQNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = None  # Will be initialized on first forward pass
        self.action_dim = action_dim

    def forward(self, grid, pos):
        x = self.conv(grid)             # shape: [B, C, H, W]
        x = x.view(x.size(0), -1)       # flatten conv output
        if self.fc is None:
            in_features = x.size(1) + pos.size(1)
            print(f"[INIT] Initializing FC with in_features = {in_features}")
            self.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim)
            )
        x = torch.cat([x, pos], dim=1)  # concat position
        return self.fc(x)

class DqnAgent(BaseAgent):
    def __init__(self,
                 env=None,
                 grid_shape=(10, 10),
                 action_dim=4,
                 buffer_size=100000,
                 batch_size=64,
                 gamma=0.99,
                 lr=1e-4,
                 epsilon=0.9,
                 target_update_freq=1000):

        self.grid_height, self.grid_width = grid_shape  # shape: (H, W)
        self.action_dim = action_dim

        self.q_net = SpatialQNetwork(action_dim)
        self.target_net = SpatialQNetwork(action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.train_steps = 0

    def prepare_inputs(self, pos, grid):
        grid = grid.copy()
        x = np.clip(pos[1], 0, self.grid_width)
        y = np.clip(pos[0], 0, self.grid_height)
        pos_tensor = torch.tensor([[x / self.grid_width, y / self.grid_height]], dtype=torch.float32)
        grid_tensor = torch.tensor(grid[np.newaxis, np.newaxis, ...] / 3.0, dtype=torch.float32)
        return grid_tensor, pos_tensor

    def take_action(self, state, grid):
        grid = grid.copy()
        grid_tensor, pos_tensor = self.prepare_inputs(state, grid)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            q_values = self.q_net(grid_tensor, pos_tensor)
            return q_values.argmax().item()

    def update(self, state, grid, reward, action, next_state, next_grid, done):
        self.buffer.append((state, grid.copy(), reward, action, next_state, next_grid.copy(), done))

    def update_batch(self, num_updates=10):
        if len(self.buffer) < self.batch_size:
            return

        for _ in range(num_updates):
            batch = random.sample(self.buffer, self.batch_size)
            states, grids, rewards, actions, next_states, next_grids, dones = zip(*batch)

            grid_tensors = torch.stack([
                torch.tensor(g[np.newaxis, ...] / 3.0, dtype=torch.float32) for g in grids
            ])
            next_grid_tensors = torch.stack([
                torch.tensor(g[np.newaxis, ...] / 3.0, dtype=torch.float32) for g in next_grids
            ])
            pos_tensors = torch.tensor([[s[1] / self.grid_width, s[0] / self.grid_height] for s in states], dtype=torch.float32)
            next_pos_tensors = torch.tensor([[s[1] / self.grid_width, s[0] / self.grid_height] for s in next_states], dtype=torch.float32)

            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            q_values = self.q_net(grid_tensors, pos_tensors).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_net(next_grid_tensors, next_pos_tensors).max(1)[0].unsqueeze(1)
                targets = rewards + self.gamma * (1 - dones) * next_q_values

            loss = self.loss_fn(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
