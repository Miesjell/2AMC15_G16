import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
import random
from torch.utils.data import DataLoader, TensorDataset

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return logits, value


class PpoNewAgent(BaseAgent):
    def __init__(self,
                 env=None,
                 state_dim=6,
                 action_dim=4,
                 gamma=0.99,
                 clip_eps=0.2,
                 lr=1e-4,
                 entropy_coef=0.05,
                 value_coef=0.5,
                 max_grad_norm=0.5,
                 seed=0,
                 hidden_size=64,
                 ppo_epochs=4,
                 batch_size=64,
                 gae_lambda=0.95, 
                 batch_steps=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.batch_steps = batch_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ActorCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def take_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.squeeze().detach())

        return action.item()

    def update(self, state, reward, action, done=False):
        self.rewards.append(reward / 10.0)  # Normalize reward
        self.dones.append(done)
        self.batch_steps += 1

        if self.batch_steps >= self.batch_size or done:
            self.finish_episode()
            self.batch_steps = 0

    def finish_episode(self):
        if len(self.rewards) == 0:
            return

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        log_probs = torch.tensor(self.log_probs).to(self.device)
        values = torch.tensor(self.values).to(self.device)
        dones = np.array(self.dones)

        with torch.no_grad():
            _, next_value = self.policy(torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device))
            next_value = next_value.item()

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            returns[t] = rewards[t] + self.gamma * next_value
            next_value = returns[t]

        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        dataset = TensorDataset(states_tensor, actions_tensor, returns_tensor, advantages_tensor, log_probs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for batch_states, batch_actions, batch_returns, batch_advantages, batch_old_log_probs in loader:
                logits, values_pred = self.policy(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                #value_loss = nn.MSELoss()(values_pred.squeeze(), batch_returns)
                value_loss = nn.MSELoss()(values_pred.view(-1), batch_returns.view(-1))

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.reset()
