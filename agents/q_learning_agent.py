"""Q Learning Agent

This is an agent that does q learning stuff.
"""

from random import randint, uniform
import numpy as np
from agents import BaseAgent


# class QLearningAgent(BaseAgent):
#     """Agent that performs a random action every time. """
#
#     def __init__(self, gamma: float = 0.98, epsilon: float = 0.01):
#         super().__init__()
#         self.dct = {}
#
#     def update(self, state: tuple[int, int], reward: float, action):
#         if state not in self.dct.keys():
#             self.dct[state] = 0
#         else:
#             self.dct[state] += 1
#
#         self.dct[state] = self.dct[state] % 4
#
#     def take_action(self, state: tuple[int, int]) -> int:
#         if state not in self.dct.keys():
#             return 0
#         else:
#             return self.dct[state]


class QLearningAgent(BaseAgent):
    """Agent that performs a random action every time."""

    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0,
    ):
        super().__init__()
        self.env = env
        self.state_space = self._get_state_space()
        self.q_table = {
            state: {action: 0.0 for action in range(4)} for state in self.state_space
        }
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def _get_state_space(self):
        """Get the state"""

        grid = np.load(self.env.grid_fp)
        rows = grid.shape[0]
        cols = grid.shape[1]

        state_space = [(x, y) for x in range(rows) for y in range(cols)]
        return state_space

    def take_action(self, state: tuple[int, int]) -> int:
        if uniform(0, 1) < self.epsilon:
            print("Random move!")
            return randint(0, 3)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

        while not done:
            action = self.take_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.update(state, reward, action, next_state)
            state = next_state

    def update(self, state: tuple[int, int], reward: float, action):
        next_state, reward, done, _ = self.env.step(action)
        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * max_next_q - self.q_table[state][action]
        )
