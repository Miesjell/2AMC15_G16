"""Continuous Agent Base.

Base class for agents that operate in continuous state spaces.
"""
from abc import ABC, abstractmethod
import numpy as np


class ContinuousBaseAgent(ABC):
    def __init__(self):
        """Base agent for continuous environments.
        
        All agents operating in continuous state spaces should inherit from this class.
        """
        pass

    @abstractmethod
    def take_action(self, state: tuple[float, float]) -> int:
        """Select an action given the current continuous state.

        Args:
            state: The current position of the agent as (x, y) continuous coordinates.
            
        Returns:
            Integer action in range [0, 7] representing one of 8 directions.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, state: tuple[float, float], reward: float, action: int, 
               next_state: tuple[float, float], done: bool):
        """Update the agent given experience.

        Args:
            state: The previous position of the agent.
            reward: The reward received.
            action: The action that was taken.
            next_state: The new position of the agent.
            done: Whether the episode terminated.
        """
        raise NotImplementedError
