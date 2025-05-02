"""Null Agent.

An agent which does nothing.
"""
import numpy as np

from agents import BaseAgent


class NullAgent(BaseAgent):

    def take_action(self, state: tuple[int, int]) -> int:
        return 4 # Changed 4 to -1 to see if that results in no error but still no action; same problem remains
    
    def update(self, state: tuple[int, int], reward: float, action):
        pass