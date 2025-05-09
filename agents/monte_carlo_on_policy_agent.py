"""Monte Carlo On-Policy Agent
This is an agent that uses on polic first visit Monte Carlo control for epsilon soft policies to approximate the optimal policy.
"""
import numpy as np
from agents import BaseAgent


# This algorithm is based on Sutton and Barto, 2nd edition, section 5.4 p.101 http://incompleteideas.net/book/RLbook2020.pdf
class MonteCarloOnPolicyAgent(BaseAgent):
    def __init__(self, gamma: float = 0.9, epsilon: float = 0.9, min_epsilon = 0.05, decay =0.999, action_space: list = None, debug: bool = False):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.state_action_values = {}  # Q(s, a)
        self.action_space = action_space if action_space is not None else [0, 1, 2, 3]
        # self.returns = {}  # Returns(s, a)
        self.returns_count = {}  # Returns count(s, a)
        self.policy = {}  # π(a|s)
        self.gamma = gamma
        self.debug = debug # Discount factor for sum of future rewards
        self.q_deltas = []
        self.log_position_behavior = True
    
    def _initialize_state_policy(self, state):
        """
        Initialize epsilon-soft policy for a new state.
        """
        if state not in self.policy:
            self.policy[state] = {}
            num_actions = len(self.action_space)
            for action in self.action_space:
                self.policy[state][action] = 1.0 / num_actions  # uniform initially, which is larger than epsilon/num_actions for any episolon smaller than 1.0 which thus satisfies an epsilon-soft policy

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

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

            if (s, a) not in visited:
                visited.add((s, a))
                self._initialize_state_policy(s)

                # Q-update with incremental mean
                key = (s, a)
                old_q = self.state_action_values.get(key, 0.0)
                self.returns_count[key] = self.returns_count.get(key, 0) + 1
                alpha = 1 / self.returns_count[key]
                self.state_action_values[key] = old_q + alpha * (G - old_q)

                # Track delta for convergence debugging
                delta = abs(old_q - self.state_action_values[key])
                self.q_deltas.append(delta)

                # ε-soft policy improvement
                best_action = max(self.action_space, key=lambda act: self.state_action_values.get((s, act), 0.0))
                num_actions = len(self.action_space)
                for act in self.action_space:
                    if act == best_action:
                        self.policy[s][act] = 1 - self.epsilon + self.epsilon / num_actions
                    else:
                        self.policy[s][act] = self.epsilon / num_actions

            # if (s, a) not in visited:
            #     visited.add((s, a))

            #     if (s, a) not in self.returns:
            #         self.returns[(s, a)] = []
            #     self.returns[(s, a)].append(G)

            #     old_q_value = self.state_action_values.get((s, a), None)
            #     self.state_action_values[(s, a)] = np.mean(self.returns[(s, a)])

            #     # if self.debug and (old_q_value is None or abs(old_q_value - self.state_action_values[(s, a)]) > 0.1):
            #     #     print(f"Q({s}, {a}) updated from {old_q_value} to {self.state_action_values[(s, a)]}")

            #     self._initialize_state_policy(s)

            #     best_action = max(
            #         self.action_space,
            #         key=lambda act: self.state_action_values.get((s, act), 0)
            #     )

            #     num_actions = len(self.action_space)
            #     for act in self.action_space:
            #         if act == best_action:
            #             self.policy[s][act] = 1 - self.epsilon + self.epsilon / num_actions
            #         else:
            #             self.policy[s][act] = self.epsilon / num_actions

            #     # if self.debug:
            #     #     print(f"Updated policy for state {s}: {self.policy[s]}")
              

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Choose an action for the given state using the current policy.
        """
        self._initialize_state_policy(state)
        actions, probs = zip(*self.policy[state].items())
        return np.random.choice(actions, p=probs)
    
    def has_converged(self, threshold=0.001, window=100):
        if len(self.q_deltas) < window:
            return False
        return np.mean(self.q_deltas[-window:]) < threshold

    def analyze_behavior(self, episode):
        if self.log_position_behavior:
            states = [s for s, _, _ in episode]
            start = states[0]
            distances = [abs(s[0] - start[0]) + abs(s[1] - start[1]) for s in states]
            if max(distances) < 3:
                print("[Warning] Agent may be stuck near starting position.")