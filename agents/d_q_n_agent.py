import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent

def _get_state_dim(env):
    state = env.reset()
    return state.shape[0] if isinstance(state, np.ndarray) else len(state)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay_steps=500_000,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=1000,
        num_actions=4,
    ):
        super().__init__()
        self.env = env
        self.state_dim = _get_state_dim(env)
        self.action_dim = num_actions

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.update_counter = 0

        self.prev_state = None
        self.prev_action = None

    def take_action(self, state):
        self.step_count += 1
        epsilon = max(self.epsilon_min, self.epsilon_start - self.step_count / self.epsilon_decay_steps)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_vals = self.policy_net(state_tensor)
            action = int(torch.argmax(q_vals, dim=1).item())

        self.prev_state = state
        self.prev_action = action
        return action

    def update(self, state, reward, actual_action):
        if self.prev_state is None or self.prev_action is None:
            return

        self.replay_buffer.append((
            np.array(self.prev_state, dtype=np.float32),
            self.prev_action,
            float(reward),
            np.array(state, dtype=np.float32)
        ))

        self.prev_state = state
        self.update_counter += 1

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Speed contrainst
        # states = torch.FloatTensor(np.array(states)) # changed for speed
        # actions = torch.LongTensor(actions).unsqueeze(1)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1)
        # next_states = torch.FloatTensor(next_states)

        states = torch.from_numpy(np.stack(states)).float()
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).float()   

        curr_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q

        loss = nn.SmoothL1Loss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset(self):
        self.prev_state = None
        self.prev_action = None
