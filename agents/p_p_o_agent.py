import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        probs = self.actor(shared_out)
        value = self.critic(shared_out)
        return probs, value


class PPOAgent(BaseAgent):
    def __init__(self,
                 env,
                 state_dim=6,
                 action_dim=4,
                 gamma=0.99,
                 clip_eps=0.2,
                 lr=3e-4,
                 entropy_coef=0.01
                 ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def take_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state_tensor)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        return action.item()

    def update(self, state, reward, action):
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))

    def finish_episode(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.cat(returns).detach()

        values = torch.cat(self.values).squeeze()
        log_probs = torch.cat(self.log_probs)
        actions = torch.cat(self.actions)
        states = torch.cat(self.states)

        advantages = returns - values
        probs, value_estimates = self.policy(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - log_probs.detach()).exp()
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(value_estimates.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset()