import random
from collections import defaultdict
from agents import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, num_actions=8):
        """initialize Q-table and hyperparameters"""
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor: how important are immediate (0)/future rewards (1)
        self.epsilon = epsilon    # exploration rate
        self.num_actions = num_actions
        self.q_table = defaultdict(lambda: [0.0] * self.num_actions)
        self.prev_state = None
        self.prev_action = None

    def take_action(self, state: tuple[int, int]) -> int:
        """choose an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action = random.choice(best_actions)
        self.prev_state = state
        self.prev_action = action
        return action

    def update(self, new_state: tuple[int, int], reward: float, actual_action: int):
        """Update Q-table using the Q-learning update rule."""
        if self.prev_state is None or self.prev_action is None:
            return  

        max_future_q = max(self.q_table[new_state])
        current_q = self.q_table[self.prev_state][self.prev_action]

        # Q-learning update
        self.q_table[self.prev_state][self.prev_action] += self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )

        # reset previous state/action if episode ends
        if reward == 10:  # reaching the goal in your reward function
            self.prev_state = None
            self.prev_action = None