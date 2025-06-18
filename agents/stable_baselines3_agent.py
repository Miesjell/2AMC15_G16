import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.spaces import Box
import torch
from agents.base_agent import BaseAgent


class GymnasiumWrapper(gym.Env):
    def __init__(self, continuous_env):
        super().__init__()
        self.env = continuous_env
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Make sure we're properly passing the done signal
        terminated = done  # Episode ends when goal is reached
        truncated = False   # We don't truncate episodes in this environment
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, 'gui') and self.env.gui is not None:
            self.env.gui.close()


class StableBaselines3Agent(BaseAgent):
    def __init__(self, env, algorithm='DQN', **kwargs):
        super().__init__()
        self.wrapped_env = env

        # Improved DQN hyperparameters for sparse reward navigation
        self.model = DQN(
            'MlpPolicy',
            self.wrapped_env,
            learning_rate=3e-4,  # Reduced learning rate for stability
            buffer_size=100000,   # Smaller buffer for faster updates
            learning_starts=1000,  # Start training sooner
            batch_size=64,        # Smaller batch size
            tau=0.005,           # Soft target update
            gamma=0.99,          # Standard discount factor
            train_freq=4,        # Train every 4 steps
            gradient_steps=1,
            target_update_interval=1000,  # More frequent target updates
            exploration_fraction=0.3,     # Shorter exploration phase
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,   # Lower final exploration
            max_grad_norm=10,
            policy_kwargs=dict(
                net_arch=[256, 256],  # Smaller network for faster training
                activation_fn=torch.nn.ReLU,
            ),
            verbose=0,  # Enable verbose output
            **kwargs
        )

        self.algorithm = algorithm

    def take_action(self, state):
        action, _ = self.model.predict(state, deterministic=False)
        return int(action)

    def update(self, state, reward, actual_action):
        pass

    def reset(self):
        pass

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = DQN.load(path, env=self.wrapped_env)