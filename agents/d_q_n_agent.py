import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent


def _get_state_dim(env):
    """Infer state dimension from a single observation."""
    state = env.reset()
    if isinstance(state, np.ndarray):
        return state.shape[0]
    elif isinstance(state, tuple) or isinstance(state, list):
        return len(state)
    else:
        raise ValueError("Cannot infer state dimension from environment reset().")


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and target network.
    """
    def __init__(
        self,
        env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        num_actions: int = 4,
    ):
        super().__init__()
        # Environment and dimensions
        self.env = env
        self.state_dim = _get_state_dim(env)
        self.action_dim = num_actions

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = 500000
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        # Networks
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_counter = 0

        # For storing previous step
        self.prev_state = None
        self.prev_action = None

    def take_action(self, state):
        """
        Selects an action using epsilon-greedy policy based on Q-values.
        """
        # Dynamic linear epsilon decay
        self.step_count += 1
        eps = max(self.epsilon_min, self.epsilon - self.step_count / 500_000)
    
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
        if random.random() < eps:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_vals = self.policy_net(state_tensor)
            action = int(torch.argmax(q_vals, dim=1).item())
    
        self.prev_state = state
        self.prev_action = action
        return action


    def update(self, state, reward, actual_action):
        """
        Stores transition and performs a learning step using a batch from replay.
        state: next state after taking action
        reward: observed reward
        actual_action: action executed (unused, uses stored prev_action)
        """
        # Only learn if we have a previous state-action
        if self.prev_state is None or self.prev_action is None:
            return

        # Store transition (state, action, reward, next_state)
        self.replay_buffer.append((
            np.array(self.prev_state, dtype=np.float32),
            self.prev_action,
            float(reward),
            np.array(state, dtype=np.float32)
        ))

        self.update_counter += 1
        # Only update after enough samples
        if len(self.replay_buffer) < self.batch_size:
            # Decay epsilon even if not learning yet
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return

        # Sample a batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        # Compute current Q-values
        curr_q = self.policy_net(states).gather(1, actions)
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q

        # Compute loss
        loss = nn.MSELoss()(curr_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        """
        Reset agent-specific variables at the start of each episode.
        """
        self.prev_state = None
        self.prev_action = None