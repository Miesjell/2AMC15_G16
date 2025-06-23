import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.action_head = nn.Linear(64, output_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        return self.action_head(x), self.value_head(x)

class PPOAgent(BaseAgent):
    def __init__(
        self,
        env,
        gamma: float = 0.99,
        lam: float = 0.9,
        clip_eps: float = 0.1,
        lr: float = 1e-4,
        batch_size: int = 4096,
        update_epochs: int = 8,
        entropy_coef: float = 0.1,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        super().__init__()
        self.env = env
        self.state_dim = env.reset().shape[0]
        self.action_dim = 4

        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm


        # Networks and optimizer
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Storage for on-policy rollout
        self.buffer = []
        self.goal_reached_once = False

        self.prev_state = None
        self.prev_action = None

    def take_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.policy_net(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        # Detach log_prob so graph isn't retained
        self.last_log_prob = dist.log_prob(action).detach()
        return action.item()

    def update(self, state, reward, actual_action):
        # Store transition only if previous valid
        if self.prev_state is not None and self.prev_action is not None:
            # After first success, only store successful trajectories
            if not hasattr(self, 'episode_success') or self.episode_success:
                self.buffer.append((
                    self.prev_state,
                    self.prev_action,
                    self.last_log_prob,
                    reward,
                    state
                ))
        # Check for goal reach
        if self.prev_state is not None and hasattr(self.env, 'world_stats'):
            if self.env.world_stats.get('total_targets_reached', 0) > 0:
                self.episode_success = True
                # Once goal reached, clear buffer to keep only new successes
                self.buffer.clear()
                # Stop exploring
                self.entropy_coef = 0.0

        self.prev_state = state
        self.prev_action = actual_action

        # Perform PPO update when enough data collected
        if len(self.buffer) >= self.batch_size:
            self._ppo_update()
            self.buffer = []

    def _ppo_update(self):
        # Unpack buffer
        states, actions, old_log_probs, rewards, next_states = zip(*self.buffer)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute values
        _, values = self.policy_net(states)
        _, next_values = self.policy_net(next_states)
        values = values.squeeze()
        next_values = next_values.squeeze()

        # GAE advantage calculation
        deltas = rewards + self.gamma * next_values - values
        advantages = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            last_adv = deltas[t] + self.gamma * self.lam * last_adv
            advantages[t] = last_adv
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.update_epochs):
            logits, value_preds = self.policy_net(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(value_preds.squeeze(), returns)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.episode_success = False
