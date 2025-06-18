
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from agents.base_agent import BaseAgent


class GymnasiumWrapper(gym.Env):
    def __init__(self, continuous_env):
        super().__init__()
        self.env = continuous_env
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

    def close(self):
        if hasattr(self.env, 'gui') and self.env.gui is not None:
            self.env.gui.close()


class StableBaselines3Agent(BaseAgent):
    def __init__(self, env, algorithm='DQN', **kwargs):
        super().__init__()
        self.wrapped_env = GymnasiumWrapper(env)
        self.model = DQN(
            'MlpPolicy',
            self.wrapped_env,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=0,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=1,
            gradient_steps=10,
            target_update_interval=500,
            exploration_fraction=0.9,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            max_grad_norm=10,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
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
