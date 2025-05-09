import numpy as np
from agents import BaseAgent

# List of possible actions and their corresponding directions
ACTION_LIST = [0, 1, 2, 3]

# Define the possible actions and their corresponding directions
ACTIONS = {
    0: (0, 1),   # Down
    1: (0, -1),  # Up
    2: (-1, 0),  # Left
    3: (1, 0)    # Right
}

def __init__(self, env):
    """
    Initializes the ValueIterationAgent from an Environment object.
    
    Args:
        env: An instance of the Environment class containing the grid, sigma, etc.
    """
    super().__init__()  # Call parent class

    self.env = env
    self.grid = Grid.load_grid(env.grid).cells
    self.gamma = getattr(env, 'gamma', 0.9)  # Default to 0.9 if not defined in env
    self.theta = getattr(env, 'theta', 1e-4)  # Default to 1e-4 if not defined in env
    self.sigma = getattr(env, 'sigma', 0.1)   # Already in your env constructor

    self.V = np.zeros(env.grid.shape) 
    self.policy = np.zeros(env.grid.shape, dtype=int)  # Initialize policy to all 0's
    
    self._value_iteration()


# class ValueIterationAgentTjerk(BaseAgent):
#     def __init__(self, grid: np.ndarray, gamma=0.9, theta=1e-4, sigma=0.1):
#         """
#         Initializes the ValueIterationAgent.
        
#         Args:
#             grid: The grid
#             gamma: Discount factor for future rewards
#             theta: Threshold to determine convergence of the value function
#             sigma: Probability of taking a random action
#         """
#         super().__init__()  #Calling parent class
#         self.grid = Grid.load_grid(grid).cells
#         self.gamma = gamma
#         self.theta = theta
#         self.sigma = sigma
#         self.V = np.zeros(grid.shape) 
#         self.policy = np.zeros(grid.shape, dtype=int)  # Initialize policy to all 0's
        
#         # Perform value iteration to compute optimal value functions and policy
#         self._value_iteration()

    def is_valid(self, pos):
        """
        Helper function to check if a tile is valid on the grid
        """
        x, y = pos
        # Position should be inside the grid and not in a wall (represented by 1 or 2)
        return (0 <= x < self.grid.shape[0] and 
                0 <= y < self.grid.shape[1] and 
                self.grid[x, y] != 1 and self.grid[x, y] != 2)

    def reward(self, pos):
        """
        Mimics the reward function in the environment
        """
        val = self.grid[pos]
        if val == 0:
            return -1  # Moving to a regular cell
        elif val == 3:
            return 10  # Reaching the goal
        elif val in [1, 2]:
            return -5  # Hitting a wall
        else:
            return -1  # Default penalty

    def _get_transition_probs(self, intended_action):
        """
        Return the probabilities of the intended and non-intended actions based on sigma.
        """
        other_actions = [a for a in ACTION_LIST if a != intended_action]
        p_intended = 1 - self.sigma  # Probability of taking intended action
        p_other = self.sigma / len(other_actions)  # Probability of taking each other action
        return [(intended_action, p_intended)] + [(a, p_other) for a in other_actions]

    def _value_iteration(self):
        """
        Perform value iteration to compute the optimal value function (V) and policy.
        This method iteratively updates the value function and policy until it has converged enough.
        """
        while True:
            delta = 0  # Track the largest change in the value function during this iteration
            
            # Iterate through each position in the grid
            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    if self.grid[x, y] in [1, 2]:  # Skip walls or obstacles
                        continue
                    
                    best_value = float('-inf')  # Initialize best value to negative infinity
                    best_action = 0  # Initialize best action (arbitrary starting action)

                    # Iterate over all possible actions to find the best one
                    for action in ACTION_LIST:
                        expected_value = 0.0
                        
                        # Compute expected value for the current action, considering random actions
                        for actual_action, prob in self._get_transition_probs(action):
                            dx, dy = ACTIONS[actual_action]  # Get the delta movement for this action to be used later
                            nx, ny = x + dx, y + dy  # Calculate new position based on action
                            
                            # Check if the new position is valid
                            if self.is_valid((nx, ny)):
                                r = self.reward((nx, ny))  # Reward for the new position and get new value
                                v = self.V[nx, ny]
                            else:
                                r = self.reward((x, y))  # Stay in the current position if invalid and use that value
                                v = self.V[x, y]

                            # Accumulate the expected value considering transition probabilities
                            expected_value += prob * (r + self.gamma * v)

                        # Maximize over all actions, resulting in the best action
                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = action

                    # Update the value function for this state
                    delta = max(delta, abs(self.V[x, y] - best_value))
                    self.V[x, y] = best_value  # Update the value for the state
                    self.policy[x, y] = best_action  # Update the policy with the best action

            # Stop if the value function has converged, the difference between this value and the last is small enough
            if delta < self.theta:
                break

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Returns the action the agent will take based on the derrived policy
        """
        return self.policy[state]

    def update(self, state: tuple[int, int], reward: float, action: int):
        """
        This function can be passed for value iteration, as the agent does not check its surroundings again
        """
        pass
