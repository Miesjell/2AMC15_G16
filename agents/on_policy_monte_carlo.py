import numpy as np
import random
from agents.base_agent import BaseAgent

class OnPolicyMonteCarlo(BaseAgent):
    def __init__(self,
                 gamma=0.9,
                 epsilon=1.0,
                 min_epsilon=0.05,
                 decay_rate=0.997,
                 freeze_exploration_after=5000, 
                 max_episode_len=3000,
                 optimistic_init=100.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.freeze_exploration_after = freeze_exploration_after
        self.max_episode_len = max_episode_len
        self.optimistic_init = optimistic_init

        # Q, returns_count
        self.Q = {}               # Q[state] = np.array([a0,...,a3])
        self.returns_count = {}   # returns_count[state][action] = visits
        self.episode = []
        self.steps_done = 0

    def take_action(self, state):
        state = tuple(state)
        self._ensure_state(state)
        self.steps_done += 1

        # ε schedule
        # Only decay until freeze point
        if self.steps_done < self.freeze_exploration_after:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
        else:
            self.epsilon = 0.0

        # ε-greedy
        if random.random() < self.epsilon:
            return random.randrange(4)
        return int(np.argmax(self.Q[state]))

    # def update(self, state, reward, action, info):
    #     # Record the experience
    #     self.episode.append((tuple(state), action, reward))

    #     # If end of episode (max length or terminal), update Q from the full episode
    #     if (len(self.episode) >= self.max_episode_len
    #         or info.get("target_reached", False)
    #         or info.get("terminated", False)):
    #         self._every_visit_update()
    #         self.episode = []

    def update(self, state, reward, action, info):
        self.episode.append((tuple(state), action, reward))

    def finalize_episode(self, info):
        if (len(self.episode) >= self.max_episode_len
            or info.get("target_reached", False)
            or info.get("terminated", False)):
            self._every_visit_update()
            self.episode = []

    def _every_visit_update(self):
        G = 0
        # Walk backward through the episode
        for state, action, reward in reversed(self.episode):
            G = self.gamma * G + reward
            self._ensure_state(state)
            # Incremental update rule (alpha = 1/N)
            self.returns_count[state][action] += 1
            alpha = 1.0 / self.returns_count[state][action]
            self.Q[state][action] += alpha * (G - self.Q[state][action])

    def _ensure_state(self, state):
        if state not in self.Q:
            # Optimistic initialization encourages exploration
            self.Q[state] = np.ones(4, dtype=float) * self.optimistic_init
            self.returns_count[state] = np.zeros(4, dtype=int)

    def print_greedy_policy(self):
        dirs = ['↓','↑','←','→']
        return {s: dirs[int(np.argmax(self.Q[s]))] for s in self.Q}
    
    def freeze_policy(self):
        """Disable exploration during evaluation."""
        self.epsilon = 0.0
