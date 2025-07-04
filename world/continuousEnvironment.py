import numpy as np
from pathlib import Path
from warnings import warn
from world.grid import Grid
from world.gui import GUI
from world.path_visualizer import visualize_path
from world.helpers import save_results, action_to_direction
from time import time, sleep
from datetime import datetime
import random

class ContinuousEnvironment:
    def __init__(
        self,
        grid_fp: Path,
        no_gui: bool = False,
        sigma: float = 0.0,
        agent_start_pos: tuple[int, int] = None,
        reward_fn: callable = None,
        target_fps: int = 30,
        random_seed=None,
        agent_size: float = 1.0,
        step_size: float = 1.0,
    ):
        random.seed(random_seed)
        self.grid_fp = grid_fp
        self.grid = Grid.load_grid(self.grid_fp).cells
        self.agent_start_pos = agent_start_pos
        self.agent_size = agent_size
        self.step_size = step_size
        self.terminal_state = False
        self.sigma = sigma

        self.no_gui = no_gui
        self.target_spf = 0.0 if target_fps <= 0 else 1.0 / target_fps
        self.gui = None

        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

    def _initialize_agent_pos(self):
        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                self.agent_pos = np.array([pos[0], pos[1]], dtype=np.float32)
            else:
                raise ValueError("Cannot place agent on obstacle or target.")
        else:
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = np.array([zeros[0][idx] + 0.5, zeros[1][idx] + 0.5], dtype=np.float32)

    def reset(self, **kwargs) -> np.ndarray:
        for k, v in kwargs.items():
            if k == "grid_fp":
                self.grid_fp = v
            elif k == "agent_start_pos":
                self.agent_start_pos = v
            elif k == "no_gui":
                self.no_gui = v
            elif k == "target_fps":
                self.target_spf = 1.0 / v

        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.visited = set()

        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        elif self.gui is not None:
            self.gui.close()

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        H, W = self.grid.shape
        x, y = self.agent_pos
        norm_x = x / (H - 1)
        norm_y = y / (W - 1)

        d = self.distance_sensor(self.grid, (x, y))
        max_dist = float(H + W)
        up = np.clip(d["up"], 0, max_dist) / max_dist
        down = np.clip(d["down"], 0, max_dist) / max_dist
        left = np.clip(d["left"], 0, max_dist) / max_dist
        right = np.clip(d["right"], 0, max_dist) / max_dist

        return np.array([norm_x, norm_y, up, down, left, right], dtype=np.float32)

    def step(self, action: int):
        if random.random() > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 3)
        direction = action_to_direction(actual_action)
        new_pos = self.agent_pos + self.step_size * np.array(direction)

        reward, goal_reached = self.reward_fn(self.grid, new_pos, self.agent_size)
        self.terminal_state = goal_reached
        self._move_agent(new_pos)

        return self._get_obs(), reward, self.terminal_state, {
            "actual_action": actual_action,
            "target_reached": goal_reached
        }

    def _move_agent(self, new_pos):
        grid_pos = tuple(np.floor(new_pos).astype(int))
        cell = self.grid[grid_pos]
        if cell in [1, 2]:
            return 
        self.agent_pos = new_pos
        if cell == 3:
            self.grid[grid_pos] = 0

    @staticmethod
    def distance_sensor(grid, agent_pos):
        row, col = agent_pos
        distances = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
        step = 0.1
        r = row - step
        while int(r) >= 0 and grid[int(r), int(col)] == 0:
            distances["up"] += step
            r -= step
        r = row + step
        while int(r) < grid.shape[0] and grid[int(r), int(col)] == 0:
            distances["down"] += step
            r += step
        c = col - step
        while int(c) >= 0 and grid[int(row), int(c)] == 0:
            distances["left"] += step
            c -= step
        c = col + step
        while int(c) < grid.shape[1] and grid[int(row), int(c)] == 0:
            distances["right"] += step
            c += step
        return distances

    def _default_reward_function(self,grid, agent_pos, agent_size) -> tuple[float, bool]:
        # Check for the agent size the corners of the bounding box
        half = agent_size / 2.0
        corners = [
            (agent_pos[0] - half, agent_pos[1] - half),
            (agent_pos[0] - half, agent_pos[1] + half),
            (agent_pos[0] + half, agent_pos[1] - half),
            (agent_pos[0] + half, agent_pos[1] + half),
        ]
        for corner in corners:
            gp = tuple(np.floor(corner).astype(int))
            cell = grid[gp]
            if cell in (1, 2):
                return -5.0, False
            if cell == 3:
                return 100.0, True
    
        # Standard penality for making a move
        reward = -0.01
    
        # Having an bonus for moving to an open space
        d = ContinuousEnvironment.distance_sensor(grid, agent_pos)
        max_view = float(grid.shape[0] + grid.shape[1])
        opening_bonus = (d["up"] + d["down"] + d["left"] + d["right"]) / (4.0 * max_view)
        reward += 0.2 * opening_bonus

        # False statement is used for checking if the target is reached
        return reward, False

    @staticmethod
    def evaluate_agent(grid_fp, agent, max_steps, sigma=0.0, agent_start_pos=None, random_seed=0, show_images=False):
        env = ContinuousEnvironment(
            grid_fp=grid_fp,
            no_gui=True,
            sigma=sigma,
            agent_start_pos=agent_start_pos,
            target_fps=-1,
            random_seed=random_seed,
            agent_size=1,
        )
        state = env.reset()
        initial = np.copy(env.grid)
        path = [env.agent_pos]
        total = 0.0
        target_reached = False

        for _ in range(max_steps):
            action = agent.take_action(state)
            state, reward, done, info = env.step(action)
            total += reward
            path.append(env.agent_pos)
            if info.get("target_reached", False):
                target_reached = True
            if done:
                break

        env.world_stats = {
            "cumulative_reward": total,
            "total_steps": len(path),
            "total_agent_moves": len(path),
            "total_failed_moves": 0,
            "total_targets_reached": int(target_reached),
        }
        img = visualize_path(initial, path)
        fname = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        save_results(fname, env.world_stats, img, show_images)
        return total