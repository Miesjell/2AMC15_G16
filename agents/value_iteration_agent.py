import random
from collections import defaultdict
from agents import BaseAgent

class ValueIterationAgent(BaseAgent):
    def __init__(self, gamma=0.9, theta=0.001, num_actions=4):
        """Initialize value table and hyperparameters"""
        super().__init__()
        self.gamma = gamma   #discount factor
        self.theta = theta   #convergence threshold
        self.num_actions = num_actions
        self.values = defaultdict(float)  
        self.policy = defaultdict(int)    
        self.model = defaultdict(list)   
        self.transitions = defaultdict(list) 
        self.states = set()
        self.prev_state = None


    def value_iteration(self):
        delta = 0
        for s in self.states:
            old = self.values[s]
            q_vals = []
            for a in range(self.num_actions):
                q = 0
                for (ns, r, p) in self.transitions.get((s,a), []):
                    q += p * (r + self.gamma * self.values[ns])
                q_vals.append(q)
            if q_vals:
                best = max(q_vals)
                self.values[s] = best
                self.policy[s] = q_vals.index(best)
                delta = max(delta, abs(old - best))
        return delta

    def take_action(self, state: tuple[int, int]) -> int:
        """Choose an action based on current policy"""
        return self.policy.get(state, random.randint(0, self.num_actions - 1))

    def update(self, new_state, reward, actual_action):
        if self.prev_state is not None:
            sa = (self.prev_state, actual_action)

            self.transitions[sa].append((new_state, reward, 1.0))
            self.states.add(self.prev_state)
            self.states.add(new_state)
            for _ in range(10):
                if self.value_iteration() < self.theta:
                    break
        self.prev_state = new_state
        

