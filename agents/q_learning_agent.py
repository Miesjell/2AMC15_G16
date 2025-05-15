import numpy as np
import random
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=0.8, # for default use
        epsilon=0.8,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        num_actions=4,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.q_table = {}

        self.prev_state = None
        self.prev_action = None

    def _get_q_values(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def take_action(self, state):
        state = tuple(state)
        q_values = self._get_q_values(state)

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(q_values))

        self.prev_state = state
        self.prev_action = action
        return action

    def update(self, state, reward, actual_action):
        if self.prev_state is None or self.prev_action is None:
            return

        state = tuple(state)
        prev_q = self._get_q_values(self.prev_state)
        curr_q = self._get_q_values(state)

        best_future_q = np.max(curr_q)
        td_target = reward + self.gamma * best_future_q
        td_error = td_target - prev_q[self.prev_action]
        prev_q[self.prev_action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
