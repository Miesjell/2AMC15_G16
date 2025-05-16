import numpy as np
import random
from agents.base_agent import BaseAgent
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

# This is a Monte Carlo Agent which implementation is bsed on Sutton and Barto's book "Reinforcement Learning: An Introduction" (2nd edition) page 101, but we do not keep track of an explicit policy here.
class MonteCarloAgent(BaseAgent):
    def __init__(
        self,
        env,
        n_actions: int = 4,
        epsilon: float = 1, # Start initially with epsilon = 1 such that we have high exploration
        gamma: float = 0.99, # for default use
        first_visit: bool = True,
    ):
        self.env = env
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.first_visit = first_visit

        self.Q: Dict[Tuple[int, int], float] = defaultdict(float) # Default dict allows us to set the default value to 0 for unseen state, action tuple
        self.returns: Dict[Tuple[int, int], List[float]] = defaultdict(list) # For each state-action pair, returns 

    def take_action(self, state: int) -> int:
        """This function selects an action based on an epsilon-greedy policy implicitly. This is done through determining if a random move should be made.
        This is an implicit way of sampling from the policy for us."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        return int(np.argmax(q_values))
    
    def update(self, episode: List[Tuple[int, int, float]]):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            if self.first_visit and (state, action) in visited:
                continue
            visited.add((state, action))
            self.returns[(state, action)].append(G)
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])

    def __str__(self):
        return "MC_Agent"

   
