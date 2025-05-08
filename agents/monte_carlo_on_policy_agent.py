import numpy as np
import random
from agents.base_agent import BaseAgent

class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, actions=[0, 1, 2, 3], epsilon=0.5, gamma=0.6, max_episode_len=400):
        """
        On-Policy Monte Carlo Agent using first-visit method.

        Set hyperparameters here before each experiment.

        """
        self.actions = actions
        self.epsilon = epsilon      # ← change here for each experiment
        self.gamma = gamma          # ← change here for each experiment
        self.max_episode_len = max_episode_len  # ← change here too

        self.Q = {}                 # Q[(state, action)] = value
        self.returns_sum = {}       # Sum of returns per (state, action)
        self.returns_count = {}     # Count of returns per (state, action)
        self.episode = []           # List of (state, action, reward)

    def take_action(self, obs, info=None):
        """Select action using epsilon-greedy policy."""
        state = tuple(obs)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_vals = [self.Q.get((state, a), 0) for a in self.actions]
            return self.actions[np.argmax(q_vals)]

    def update(self, obs, reward, actual_action, info=None):
        """Store experience in the current episode."""
        state = tuple(obs)
        self.episode.append((state, actual_action, reward))

    def update_Q(self):
        """At the end of the episode, update Q-values using first-visit MC."""
        G = 0
        visited = set()

        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))

                self.returns_sum[(state, action)] = self.returns_sum.get((state, action), 0) + G
                self.returns_count[(state, action)] = self.returns_count.get((state, action), 0) + 1
                self.Q[(state, action)] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

        self.episode = []

    def __str__(self):
        return f"MonteCarloOnPolicyAgent(epsilon={self.epsilon}, gamma={self.gamma})"
