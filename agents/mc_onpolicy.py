"""Monte Carlo On-Policy Agent
This is an agent that uses Monte Carlo methods to learn the value function.
"""
import numpy as np
from agents import BaseAgent

class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, gamma: float = 0.9, epsilon: float = 0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_action_values = {}  # Q(s, a)
        self.returns = {}  # Returns(s, a)
        self.policy = {}  # π(a|s)
        # Could be a place to specify the number of episodes to run

    def update(self, state: tuple[int, int], reward: float, action, episode: list):
        """
        Update the state-action value function and policy.
        Args:
            state: The current state.
            reward: The reward received.
            action: The action taken.
            episode: The full episode as a list of (state, action, reward) tuples.
        """
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + self.gamma * G  # Discounted sum of rewards

            if (s, a) not in [(x[0], x[1]) for x in episode[:t]]:
                if (s, a) not in self.returns:
                    self.returns[(s, a)] = []
                self.returns[(s, a)].append(G)

                # Update Q(s, a)
                self.state_action_values[(s, a)] = np.mean(self.returns[(s, a)])

                # Update policy π(a|s)
                best_action = max(
                    [a for a in range(4)],
                    key=lambda a: self.state_action_values.get((s, a), 0)
                )
                for a in range(4):
                    if a == best_action:
                        self.policy[(s, a)] = 1 - self.epsilon + (self.epsilon / 4)
                    else:
                        self.policy[(s, a)] = self.epsilon / 4

    def take_action(self, state: tuple[int, int]) -> int:
        # Choose an action based on the policy
        if state not in self.policy:
            # Initialize uniform random policy if state is unseen
            self.policy[state] = {a: 1 / 4 for a in range(4)} # the probablities for each action all become 0.25
        actions, probabilities = zip(*self.policy[state].items())
        return np.random.choice(actions, p=probabilities)