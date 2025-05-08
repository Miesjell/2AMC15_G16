"""Monte Carlo On-Policy Agent
This is an agent that uses on polic first visit Monte Carlo control for epsilon soft policies to approximate the optimal policy.
"""
import numpy as np
from agents import BaseAgent


# This algorithm is based on Sutton and Barto, 2nd edition, section 5.4 p.101 http://incompleteideas.net/book/RLbook2020.pdf
class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, gamma: float = 0.9, epsilon: float = 0.3, action_space: list = None):
        # Need to inialize an epsilon greedy policy, a Q fuction and a returns list
        self.epsilon = epsilon
        self.state_action_values = {}  # Q(s, a)
        self.action_space = action_space if action_space is not None else [0, 1, 2, 3]
        self.returns = {}  # Returns(s, a)
        self.policy = {}  # π(a|s)
        self.gamma = gamma # Discount factor for sum of future rewards
        # Explicity initalize the greedy policy here
        # Could be a place to specify the number of episodes to run

    def _initialize_state_policy(self, state):
        """
        Initialize epsilon-soft policy for a new state.
        """
        if state not in self.policy:
            self.policy[state] = {}
            num_actions = len(self.action_space)
            for action in self.action_space:
                self.policy[state][action] = 1.0 / num_actions  # uniform initially, which is larger than epsilon/num_actions for any episolon smaller than 1.0 which thus satisfies an epsilon-soft policy

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
        visited = set() # Empty set

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = r + self.gamma * G

            if (s, a) not in visited: # First visit only
                visited.add((s, a))

                if (s, a) not in self.returns: # From other episodes
                    self.returns[(s, a)] = []
                self.returns[(s, a)].append(G)

                # Update Q(s, a)
                self.state_action_values[(s, a)] = np.mean(self.returns[(s, a)])

                # This only does something when the state is not already in the policy
                self._initialize_state_policy(s)

                # Update policy π(a|s) to be epsilon-greedy
                best_action = max(
                    self.action_space,
                    key=lambda act: self.state_action_values.get((s, act), 0) # zero to ensure non-existing actions are not considered
                )

                num_actions = len(self.action_space)
                for act in self.action_space:
                    if act == best_action:
                        self.policy[s][act] = 1 - self.epsilon + self.epsilon / num_actions
                    else:
                        self.policy[s][act] = self.epsilon / num_actions

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Choose an action for the given state using the current policy.
        """
        self._initialize_state_policy(state)
        actions, probs = zip(*self.policy[state].items())
        return np.random.choice(actions, p=probs)