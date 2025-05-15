import numpy as np
from agents import BaseAgent
from world.grid import Grid  # If you're using Grid.load_grid()

# Define directions for actions
ACTION_LIST = [0, 1, 2, 3]
ACTIONS = {
    0: (0, 1),   # Down
    1: (0, -1),  # Up
    2: (-1, 0),  # Left
    3: (1, 0)    # Right
}

class ValueIterationAgent(BaseAgent):
    def __init__(self, env):
        """
        Initializes the ValueIterationAgent using an Environment instance.
        """
        
        super().__init__()
        self.env = env
        self.grid = Grid.load_grid(env.grid_fp).cells
        self.gamma = getattr(env, 'gamma', 0.8)
        self.theta = getattr(env, 'theta', 1e-4)
        self.sigma = getattr(env, 'sigma', 0.2)
        self.V = np.zeros(self.grid.shape)
        self.policy = np.zeros(self.grid.shape, dtype=int)
        self._value_iteration()

    def is_valid(self, pos):
        x, y = pos
        return (0 <= x < self.grid.shape[0] and 
                0 <= y < self.grid.shape[1] and 
                self.grid[x, y] != 1 and self.grid[x, y] != 2)

    def reward(self, pos):
        val = self.grid[pos]
        if val == 0:
            return -1
        elif val == 3:
            return 10
        elif val in [1, 2]:
            return -5
        return -1

    def _get_transition_probs(self, intended_action):
        other_actions = [a for a in ACTION_LIST if a != intended_action]
        p_intended = 1 - self.sigma
        p_other = self.sigma / len(other_actions)
        return [(intended_action, p_intended)] + [(a, p_other) for a in other_actions]

    def _value_iteration(self):
        while True:
            delta = 0
            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    if self.grid[x, y] in [1, 2]:
                        continue
                    best_value = float('-inf')
                    best_action = 0
                    for action in ACTION_LIST:
                        expected_value = 0.0
                        for actual_action, prob in self._get_transition_probs(action):
                            dx, dy = ACTIONS[actual_action]
                            nx, ny = x + dx, y + dy
                            if self.is_valid((nx, ny)):
                                r = self.reward((nx, ny))
                                v = self.V[nx, ny]
                            else:
                                r = self.reward((x, y))
                                v = self.V[x, y]
                            expected_value += prob * (r + self.gamma * v)
                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = action
                    delta = max(delta, abs(self.V[x, y] - best_value))
                    self.V[x, y] = best_value
                    self.policy[x, y] = best_action
            if delta < self.theta:
                break

    def take_action(self, state: tuple[int, int]) -> int:
        return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        pass
