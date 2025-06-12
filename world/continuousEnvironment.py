import random
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime

from world.helpers import save_results, action_to_direction

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    import sys, os
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path


class ContinuousEnvironment:
    def __init__(
        self,
        grid_fp: Path,
        no_gui: bool = False,
        sigma: float = 0.0,
        agent_start_pos: tuple[int, int] = None,
        reward_fn: callable = None,
        target_fps: int = 30,
        random_seed: int | float | str | bytes | bytearray | None = 0,
        agent_size: float = 1.0,
    ):
        random.seed(random_seed)

        # Load grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        self.grid_fp = grid_fp

        # Other settings
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        self.agent_size = agent_size
        # keep track of which grid‐cells we’ve been in for novelty bonus
        self.visited: set[tuple[int,int]] = set()

        # Reward function
        if reward_fn is None:
            warn("No reward function provided. Using default reward.")
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn

        # GUI
        self.no_gui = no_gui
        self.target_spf = 0.0 if target_fps <= 0 else 1.0 / target_fps
        self.gui = None

    def _reset_info(self) -> dict:
        return {"target_reached": False, "agent_moved": False, "actual_action": None}

    @staticmethod
    def _reset_world_stats() -> dict:
        return {
            "cumulative_reward": 0,
            "total_steps": 0,
            "total_agent_moves": 0,
            "total_failed_moves": 0,
            "total_targets_reached": 0,
        }

    def _initialize_agent_pos(self):
        if self.agent_start_pos is not None:
            pos = (self.agent_start_pos[0], self.agent_start_pos[1])
            if self.grid[pos] == 0:
                self.agent_pos = np.array([pos[0] + 0.5, pos[1] + 0.5], dtype=np.float32)
            else:
                raise ValueError("Cannot place agent on obstacle or target.")
        else:
            warn("No initial agent positions given. Randomly placing agent.")
            zeros = np.where(self.grid == 0)
            idx = random.randint(0, len(zeros[0]) - 1)
            self.agent_pos = np.array(
                [zeros[0][idx] + 0.5, zeros[1][idx] + 0.5], dtype=np.float32
            )

    def reset(self, **kwargs) -> np.ndarray:
        # Override settings
        for k, v in kwargs.items():
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1.0 / v
                case _:
                    raise ValueError(f"{k} is not a valid argument.")

        # Reset world
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()

        # GUI
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        elif self.gui is not None:
            self.gui.close()

        return self._get_obs()

    def _check_agent_size(self, agent_pos: tuple[float, float]) -> bool:
        half = self.agent_size / 2
        corners = [
            (agent_pos[0] - half, agent_pos[1] - half),
            (agent_pos[0] - half, agent_pos[1] + half),
            (agent_pos[0] + half, agent_pos[1] - half),
            (agent_pos[0] + half, agent_pos[1] + half),
        ]
        for corner in corners:
            grid_pos = tuple(np.floor(corner).astype(int))
            if self.grid[grid_pos] not in [0, 3]:
                return False
        return True

    def _move_agent(self, new_pos: tuple[float, float]):
        grid_pos = tuple(np.floor(new_pos).astype(int))
        cell = self.grid[grid_pos]
        if cell in [1, 2]:
            self.world_stats["total_failed_moves"] += 1
            self.info["agent_moved"] = False
        else:
            if self._check_agent_size(new_pos):
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
                if cell == 3:
                    self.grid[grid_pos] = 0
                    self.terminal_state = (np.sum(self.grid == 3) == 0)
                    self.info["target_reached"] = True
                    self.world_stats["total_targets_reached"] += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.world_stats["total_steps"] += 1

        # GUI pause/render
        is_single = False
        if not self.no_gui:
            start = time()
            while self.gui.paused:
                if self.gui.step:
                    is_single = True
                    self.gui.step = False
                    break
                paused_info = self._reset_info()
                paused_info["agent_moved"] = True
                self.gui.render(self.grid, self.agent_pos, paused_info, 0, is_single)

        # Stochastic action
        actual = action if random.random() > self.sigma else random.randint(0, 3)
        self.info["actual_action"] = actual

        # Move
        direction = action_to_direction(actual)
        step_size = 0.2
        new_pos = self.agent_pos + step_size * np.array(direction)
        reward = self.reward_fn(self.grid, new_pos, agent_size=self.agent_size)
        self._move_agent(new_pos)
        self.world_stats["cumulative_reward"] += reward

        # GUI render
        if not self.no_gui:
            wait_t = self.target_spf - (time() - start)
            if wait_t > 0:
                sleep(wait_t)
            self.gui.render(self.grid, self.agent_pos, self.info, reward, is_single)

        return self._get_obs(), reward, self.terminal_state, self.info

    @staticmethod
    def distance_sensor(grid, agent_pos):
        row, col = agent_pos
        distances = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
        step = 0.1
        # Up
        r = row - step
        while int(r) >= 0 and grid[int(r), int(col)] == 0:
            distances["up"] += step
            r -= step
        # Down
        r = row + step
        while int(r) < grid.shape[0] and grid[int(r), int(col)] == 0:
            distances["down"] += step
            r += step
        # Left
        c = col - step
        while int(c) >= 0 and grid[int(row), int(c)] == 0:
            distances["left"] += step
            c -= step
        # Right
        c = col + step
        while int(c) < grid.shape[1] and grid[int(row), int(c)] == 0:
            distances["right"] += step
            c += step
        return distances

    def _get_obs(self) -> np.ndarray:
        """Return normalized [x, y, up, down, left, right] floats."""
        H, W = self.grid.shape
        x, y = self.agent_pos

        # Normalize position
        norm_x = x / (H - 1)
        norm_y = y / (W - 1)

        # Get and normalize distances
        d = self.distance_sensor(self.grid, (x, y))
        max_dist = float(H + W)
        up    = np.clip(d["up"],    0, max_dist) / max_dist
        down  = np.clip(d["down"],  0, max_dist) / max_dist
        left  = np.clip(d["left"],  0, max_dist) / max_dist
        right = np.clip(d["right"], 0, max_dist) / max_dist

        return np.array([norm_x, norm_y, up, down, left, right], dtype=np.float32)

    
    def _default_reward_function(self, grid, agent_pos, agent_size) -> float:
        """
        Reward breakdown:
          -5.0   for collision (wall/obstacle)
         +500.0 for reaching the kitchen (target)
          -0.01  per step (time penalty)
         +0.2   opening bonus (from local 4-way sensor)
         +0.3   novelty bonus for visiting new grid cells
        """
        import numpy as np
    
        # Configurable constants
        GOAL_REWARD = 500.0
        COLLISION_PENALTY = -5.0
        STEP_PENALTY = -0.01
        OPENING_WEIGHT = 0.2
        NOVELTY_BONUS = 0.3
    
        half = agent_size / 2.0
    
        # 1) Check collisions & goal at the four corners
        corners = [
            (agent_pos[0] - half, agent_pos[1] - half),
            (agent_pos[0] - half, agent_pos[1] + half),
            (agent_pos[0] + half, agent_pos[1] - half),
            (agent_pos[0] + half, agent_pos[1] + half),
        ]
        for corner in corners:
            gp = tuple(np.floor(corner).astype(int))
            cell = grid[gp]
            if cell in (1, 2):  # wall or obstacle
                return COLLISION_PENALTY
            if cell == 3:  # kitchen
                return GOAL_REWARD
    
        # 2) Base step penalty
        reward = STEP_PENALTY
    
        # 3) Opening bonus (normalized local openness)
        d = ContinuousEnvironment.distance_sensor(grid, agent_pos)
        max_view = float(grid.shape[0] + grid.shape[1])
        opening_bonus = (d["up"] + d["down"] + d["left"] + d["right"]) / (4.0 * max_view)
        reward += OPENING_WEIGHT * opening_bonus
    
        return reward



    @staticmethod
    def evaluate_agent(
        grid_fp: Path,
        agent: BaseAgent,
        max_steps: int,
        sigma: float = 0.0,
        agent_start_pos: tuple[int, int] = None,
        random_seed: int | float | str | bytes | bytearray = 0,
        show_images: bool = False,
    ):
        env = ContinuousEnvironment(
            grid_fp=grid_fp,
            no_gui=True,
            sigma=sigma,
            agent_start_pos=agent_start_pos,
            target_fps=-1,
            random_seed=random_seed,
            agent_size=0.5,
        )
        state = env.reset()
        initial = np.copy(env.grid)
        path = [env.agent_pos]
        total = 0.0

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, reward, done, _ = env.step(action)
            total += reward
            path.append(env.agent_pos)
            if done:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)
        img = visualize_path(initial, path)
        fname = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        save_results(fname, env.world_stats, img, show_images)
        return total
