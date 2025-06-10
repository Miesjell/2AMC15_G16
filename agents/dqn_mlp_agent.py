import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from agents import BaseAgent


class MLPQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Increased network capacity
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DqnMlpAgent(BaseAgent):
    def __init__(self,
                 env=None,
                 grid_shape=(10, 10),
                 action_dim=4,
                 buffer_size=50000,  # Reduced buffer size
                 batch_size=32,  # Smaller batch size
                 gamma=0.99,
                 lr=3e-4,  # Higher learning rate
                 epsilon=1.0,
                 epsilon_min=0.05,  # Added minimum epsilon
                 epsilon_decay=0.9995,
                 target_update_freq=500):  # More frequent target updates

        self.grid_height, self.grid_width = grid_shape
        self.action_dim = action_dim
        self.state_dim = 2  # x, y

        self.q_net = MLPQNetwork(self.state_dim, self.action_dim)
        self.target_net = MLPQNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # Changed to MSE loss

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_steps = 0

    def normalize_state(self, state):
        """Normalize state to [0, 1] range"""
        x, y = state
        # Add small epsilon to avoid division by zero
        return np.array([
            x / max(self.grid_width - 1, 1),
            y / max(self.grid_height - 1, 1)
        ], dtype=np.float32)

    def take_action(self, state, _grid=None):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(self.normalize_state(state), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def update(self, state, _grid, reward, action, next_state, _next_grid, done):
        """Store experience in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def update_batch(self, num_updates=1):
        """Perform batch updates on the Q-network"""
        if len(self.buffer) < self.batch_size:
            return

        total_loss = 0
        for _ in range(num_updates):
            # Sample batch from replay buffer
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.tensor([self.normalize_state(s) for s in states], dtype=torch.float32)
            next_states = torch.tensor([self.normalize_state(s) for s in next_states], dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Current Q values
            current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

            # Compute loss
            loss = self.loss_fn(current_q_values, target_q_values)
            total_loss += loss.item()

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.train_steps += 1

            # Update target network
            if self.train_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                # print(f"Target network updated at step {self.train_steps}")

        return total_loss / num_updates if num_updates > 0 else 0

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)