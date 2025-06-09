"""
Continuous Environment for Reinforcement Learning.

This environment extends the discrete grid world to a continuous state space
where the agent can move to any real-valued coordinate within the grid bounds.
"""
import random
import datetime
import numpy as np
from tqdm import trange
from pathlib import Path
from warnings import warn
from time import time, sleep
from datetime import datetime
from world.helpers import save_results
import math
from dataclasses import dataclass
from typing import Literal

try:
    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path
except ModuleNotFoundError:
    from os import path
    from os import pardir
    import sys

    root_path = path.abspath(path.join(
        path.join(path.abspath(__file__), pardir), pardir)
    )

    if root_path not in sys.path:
        sys.path.append(root_path)

    from agents import BaseAgent
    from world.grid import Grid
    from world.gui import GUI
    from world.path_visualizer import visualize_path


class GameObject:
    """Represents an object in the continuous world with position and radius."""
    
    def __init__(self, x: float, y: float, radius: float, obj_type: str):
        self.x = x
        self.y = y
        self.radius = radius
        self.type = obj_type  # 'robot', 'wall', 'target'
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)
    
    def collides_with(self, other_x: float, other_y: float, other_radius: float = 0.0) -> bool:
        """Check if this object collides with another object or point."""
        return self.distance_to(other_x, other_y) < (self.radius + other_radius)
    
    def __repr__(self):
        return f"GameObject({self.type}, pos=({self.x:.2f}, {self.y:.2f}), r={self.radius})"


def continuous_action_to_direction(action: int, step_size: float = 0.1) -> tuple[float, float]:
    """Convert discrete action to continuous direction vector.
    
    Args:
        action: Integer action (0-7 for 8 directions)
        step_size: Size of the step to take
        
    Returns:
        Tuple of (dx, dy) direction vector
    """
    # 8 directions, every 45 degrees
    # 0: Right (0°), 1: Down-Right (45°), 2: Down (90°), 3: Down-Left (135°)
    # 4: Left (180°), 5: Up-Left (225°), 6: Up (270°), 7: Up-Right (315°)
    
    angles = {
        0: 0,      # Right
        1: 45,     # Down-Right
        2: 90,     # Down
        3: 135,    # Down-Left
        4: 180,    # Left
        5: 225,    # Up-Left
        6: 270,    # Up
        7: 315,    # Up-Right
    }
    
    if action not in angles:
        raise ValueError(f"Action {action} not in valid range [0-7]")
    
    angle_rad = math.radians(angles[action])
    dx = step_size * math.cos(angle_rad)
    dy = step_size * math.sin(angle_rad)
    
    return (dx, dy)


class ContinuousEnvironment:
    def __init__(self,
                 grid_fp: Path,
                 no_gui: bool = False,
                 sigma: float = 0.,
                 agent_start_pos: tuple[float, float] = None,
                 reward_fn: callable = None,
                 target_fps: int = 30,
                 random_seed: int | float | str | bytes | bytearray | None = 0,
                 step_size: float = 0.1,
                 collision_radius: float = 0.1):
        
        """Creates the Continuous Grid Environment for Reinforcement Learning.

        Args:
            grid_fp: Path to the grid file to use.
            no_gui: True if no GUI is desired.
            sigma: The stochasticity of the environment.
            agent_start_pos: Tuple where agent should start (continuous coords).
            reward_fn: Custom reward function to use.
            target_fps: How fast the simulation should run.
            random_seed: The random seed to use.
            step_size: Size of each step the agent takes.
            collision_radius: Radius for collision detection with obstacles.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize Grid
        if not grid_fp.exists():
            raise FileNotFoundError(f"Grid {grid_fp} does not exist.")
        else:
            self.grid_fp = grid_fp

        # Initialize other variables
        self.agent_start_pos = agent_start_pos
        self.terminal_state = False
        self.sigma = sigma
        self.step_size = step_size
        self.collision_radius = collision_radius
        
        # Load the grid
        self.grid = Grid.load_grid(self.grid_fp).cells
        
        # Objects in the world
        self.objects = []  # List of GameObject instances
        
        # Set reward function
        if reward_fn is None:
            self.reward_fn = self._default_reward_function
        else:
            self.reward_fn = reward_fn
        
        # GUI specific code
        self.no_gui = no_gui
        if target_fps <= 0:
            self.target_spf = 0.
        else:
            self.target_spf = 1. / target_fps
        self.gui = None

    def _reset_info(self) -> dict:
        """Resets the info dictionary."""
        return {"target_reached": False,
                "agent_moved": False,
                "actual_action": None,
                "collision": False}
    
    @staticmethod
    def _reset_world_stats() -> dict:
        """Resets the world stats dictionary."""
        return {"cumulative_reward": 0,
                "total_steps": 0,
                "total_agent_moves": 0,
                "total_failed_moves": 0,
                "total_targets_reached": 0,
                "total_collisions": 0}

    def _initialize_agent_pos(self):
        """Initializes agent position from given location or randomly."""
        if self.agent_start_pos is not None:
            x, y = self.agent_start_pos
            # Ensure position is within bounds
            x = max(0.0, min(float(self.grid.shape[0] - 1), x))
            y = max(0.0, min(float(self.grid.shape[1] - 1), y))
            self.agent_pos = (x, y)
        else:
            # Random position in valid area
            warn("No initial agent position given. Randomly placing agent.")
            # Find empty cells and choose one randomly
            empty_cells = np.where(self.grid == 0)
            if len(empty_cells[0]) == 0:
                raise ValueError("No empty cells available for agent placement")
            
            idx = random.randint(0, len(empty_cells[0]) - 1)
            # Add small random offset within the cell
            base_x, base_y = empty_cells[0][idx], empty_cells[1][idx]
            self.agent_pos = (
                float(base_x) + random.uniform(0.1, 0.9),
                float(base_y) + random.uniform(0.1, 0.9)
            )

    def _is_position_valid(self, pos: tuple[float, float]) -> bool:
        """Check if a continuous position is valid (not in obstacle)."""
        x, y = pos
        
        # Check bounds
        if x < 0 or y < 0 or x >= self.grid.shape[0] or y >= self.grid.shape[1]:
            return False
        
        # Check collision with obstacles using circular collision detection
        grid_x, grid_y = int(x), int(y)
        
        # Check the current cell and neighboring cells for obstacles
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x, check_y = grid_x + dx, grid_y + dy
                
                # Skip if out of bounds
                if (check_x < 0 or check_y < 0 or 
                    check_x >= self.grid.shape[0] or check_y >= self.grid.shape[1]):
                    continue
                
                # If this cell contains an obstacle, check collision
                if self.grid[check_x, check_y] in [1, 2]:  # Wall or obstacle
                    # Calculate distance from agent center to obstacle cell center
                    cell_center_x, cell_center_y = check_x + 0.5, check_y + 0.5
                    distance = math.sqrt((x - cell_center_x)**2 + (y - cell_center_y)**2)
                    
                    # If too close to obstacle, collision detected
                    if distance < (0.5 + self.collision_radius):
                        return False
        
        return True

    def add_object(self, obj: GameObject):
        """Add a GameObject to the environment."""
        self.objects.append(obj)
    
    def clear_objects(self):
        """Clear all objects from the environment."""
        self.objects.clear()
    
    def create_objects_from_grid(self):
        """Create GameObject instances from the discrete grid."""
        self.clear_objects()
        
        # Create objects for each grid cell
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                cell_value = self.grid[i, j]
                center_x, center_y = i + 0.5, j + 0.5
                
                if cell_value == 1:  # Wall
                    self.add_object(GameObject(center_x, center_y, 0.4, 'wall'))
                elif cell_value == 2:  # Obstacle
                    self.add_object(GameObject(center_x, center_y, 0.3, 'obstacle'))
                elif cell_value == 3:  # Target
                    self.add_object(GameObject(center_x, center_y, 0.3, 'target'))
    
    def _is_position_valid_objects(self, pos: tuple[float, float]) -> bool:
        """Check if position is valid using GameObject collision detection."""
        x, y = pos
        
        # Check bounds
        if x < 0 or y < 0 or x >= self.grid.shape[0] or y >= self.grid.shape[1]:
            return False
        
        # Check collision with wall/obstacle objects
        for obj in self.objects:
            if obj.type in ['wall', 'obstacle']:
                if obj.collides_with(x, y, self.collision_radius):
                    return False
        
        return True
    
    def _check_target_reached_objects(self, pos: tuple[float, float]) -> bool:
        """Check if agent reached target using GameObject detection."""
        x, y = pos
        
        # Check collision with target objects
        for obj in self.objects:
            if obj.type == 'target':
                if obj.collides_with(x, y, self.collision_radius):
                    return True
        
        return False

    def get_objects_by_type(self, obj_type: str) -> list[GameObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects if obj.type == obj_type]

    def _check_target_reached(self, pos: tuple[float, float]) -> bool:
        """Check if agent has reached any target."""
        x, y = pos
        grid_x, grid_y = int(x), int(y)
        
        # Check bounds
        if (grid_x < 0 or grid_y < 0 or 
            grid_x >= self.grid.shape[0] or grid_y >= self.grid.shape[1]):
            return False
        
        # Check if current cell contains a target
        if self.grid[grid_x, grid_y] == 3:
            # Check if agent is close enough to the target center
            cell_center_x, cell_center_y = grid_x + 0.5, grid_y + 0.5
            distance = math.sqrt((x - cell_center_x)**2 + (y - cell_center_y)**2)
            return distance < 0.5  # Agent must be within the target cell
        
        return False

    def reset(self, **kwargs) -> tuple[float, float]:
        """Reset the environment to an initial state."""
        for k, v in kwargs.items():
            match k:
                case "grid_fp":
                    self.grid_fp = v
                case "agent_start_pos":
                    self.agent_start_pos = v
                case "no_gui":
                    self.no_gui = v
                case "target_fps":
                    self.target_spf = 1. / v
                case "step_size":
                    self.step_size = v
                case "collision_radius":
                    self.collision_radius = v
                case _:
                    raise ValueError(f"{k} is not a valid keyword argument.")
        
        # Reset variables (reload grid in case it was modified)
        self.grid = Grid.load_grid(self.grid_fp).cells
        self._initialize_agent_pos()
        self.terminal_state = False
        self.info = self._reset_info()
        self.world_stats = self._reset_world_stats()
        
        # Create GameObject instances from the grid
        self.create_objects_from_grid()

        # GUI specific code
        if not self.no_gui:
            self.gui = GUI(self.grid.shape)
            self.gui.reset()
        else:
            if self.gui is not None:
                self.gui.close()

        return self.get_enhanced_state()

    def _move_agent(self, new_pos: tuple[float, float]):
        """Moves the agent if possible and updates stats."""
        if self._is_position_valid(new_pos):
            # Check if target reached
            if self._check_target_reached(new_pos):
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.info["target_reached"] = True
                self.world_stats["total_agent_moves"] += 1
                self.world_stats["total_targets_reached"] += 1
                
                # Remove target from grid
                grid_x, grid_y = int(new_pos[0]), int(new_pos[1])
                self.grid[grid_x, grid_y] = 0
                
                # Check if all targets collected
                if np.sum(self.grid == 3) == 0:
                    self.terminal_state = True
            else:
                # Normal movement to empty space
                self.agent_pos = new_pos
                self.info["agent_moved"] = True
                self.world_stats["total_agent_moves"] += 1
        else:
            # Collision with obstacle or boundary
            self.info["agent_moved"] = False
            self.info["collision"] = True
            self.world_stats["total_failed_moves"] += 1
            self.world_stats["total_collisions"] += 1

    def step(self, action: int) -> tuple[tuple[float, float], float, bool, dict]:
        """Make the agent take a step in the continuous environment.
        
        Args:
            action: Integer representing the action (0-7 for 8 directions)
        
        Returns:
            0) Current state (continuous position),
            1) The reward for the agent,
            2) If the terminal state has been reached,
            3) Additional info
        """
        self.world_stats["total_steps"] += 1
        
        # Reset info
        self.info = self._reset_info()
        
        # Add stochasticity
        val = random.random()
        if val > self.sigma:
            actual_action = action
        else:
            actual_action = random.randint(0, 7)  # 8 possible actions
        
        self.info["actual_action"] = actual_action
        
        # Convert action to direction
        direction = continuous_action_to_direction(actual_action, self.step_size)
        new_pos = (self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1])
        
        # Calculate reward
        reward = self.reward_fn(self.grid, new_pos, self.agent_pos)
        
        # Move agent
        self._move_agent(new_pos)
        
        self.world_stats["cumulative_reward"] += reward

        # GUI specific code (simplified for continuous)
        if not self.no_gui and self.gui is not None:
            # Convert continuous position to discrete for GUI
            discrete_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
            self.gui.render(self.grid, discrete_pos, self.info, reward, False)

        # Create enhanced state: position + distance sensors
        enhanced_state = self.get_enhanced_state()
        
        return enhanced_state, reward, self.terminal_state, self.info

    @staticmethod
    def _default_reward_function(grid, new_pos: tuple[float, float], old_pos: tuple[float, float]) -> float:
        """Enhanced reward function with distance-based guidance.
        
        Args:
            grid: The discrete grid for reference
            new_pos: New agent position (continuous)
            old_pos: Previous agent position (continuous)
            
        Returns:
            Reward value
        """
        x, y = new_pos
        old_x, old_y = old_pos
        
        # Check bounds
        if x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1]:
            return -2.0  # Strong out of bounds penalty
        
        # Get grid cell value
        grid_x, grid_y = int(x), int(y)
        cell_value = grid[grid_x, grid_y]
        
        # Base rewards for cell type
        if cell_value == 1 or cell_value == 2:  # Wall or obstacle
            return -1.0  # Strong collision penalty
        elif cell_value == 3:  # Target
            return 50.0  # Very large reward for reaching target
        
        # For empty spaces, calculate distance-based reward
        base_reward = -0.02  # Small time penalty
        
        # Find all targets in the grid
        target_positions = np.where(grid == 3)
        if len(target_positions[0]) == 0:
            return base_reward
        
        # Calculate distances to all targets
        target_rows, target_cols = target_positions
        
        # Distance from new position to nearest target
        new_distances = np.sqrt((x - (target_rows + 0.5))**2 + (y - (target_cols + 0.5))**2)
        min_new_dist = np.min(new_distances)
        
        # Distance from old position to nearest target
        old_distances = np.sqrt((old_x - (target_rows + 0.5))**2 + (old_y - (target_cols + 0.5))**2)
        min_old_dist = np.min(old_distances)
        
        # Reward for getting closer to target
        distance_reward = (min_old_dist - min_new_dist) * 0.5
        
        # Add small bonus for being close to target
        proximity_bonus = max(0, (5.0 - min_new_dist) * 0.02)
        
        # Encourage exploration by giving small bonus for visiting new areas
        exploration_bonus = 0.001
        
        total_reward = base_reward + distance_reward + proximity_bonus + exploration_bonus
        
        return total_reward

    def get_state_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get the bounds of the continuous state space.
        
        Returns:
            ((min_x, max_x), (min_y, max_y))
        """
        return ((0.0, float(self.grid.shape[0])), (0.0, float(self.grid.shape[1])))

    def get_action_space_size(self) -> int:
        """Get the size of the discrete action space."""
        return 8  # 8 directions

    @staticmethod
    def evaluate_agent(grid_fp: Path,
                       agent: BaseAgent,
                       max_steps: int,
                       sigma: float = 0.,
                       agent_start_pos: tuple[float, float] = None,
                       random_seed: int | float | str | bytes | bytearray = 0,
                       show_images: bool = False,
                       step_size: float = 0.1):
        """Evaluates a trained agent's performance in continuous environment."""
        
        env = ContinuousEnvironment(grid_fp=grid_fp,
                                  no_gui=True,
                                  sigma=sigma,
                                  agent_start_pos=agent_start_pos,
                                  target_fps=-1,
                                  random_seed=random_seed,
                                  step_size=step_size)
        
        state = env.reset()
        initial_grid = np.copy(env.grid)
        
        # Track continuous path
        agent_path = [env.agent_pos]
        total_return = 0

        for _ in trange(max_steps, desc="Evaluating agent"):
            action = agent.take_action(state)
            state, reward, terminated, _ = env.step(action)
            
            total_return += reward
            agent_path.append(state)

            if terminated:
                break

        env.world_stats["targets_remaining"] = np.sum(env.grid == 3)
        
        # Convert continuous path to discrete for visualization
        discrete_path = [(int(pos[0]), int(pos[1])) for pos in agent_path]
        path_image = visualize_path(initial_grid, discrete_path)
        file_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        save_results(file_name, env.world_stats, path_image, show_images)
        return total_return

    def get_distance_sensors(self, position: tuple[float, float], max_range: float = 2.0) -> list[float]:
        """Get distance to nearest obstacle in each of the 8 directions.
        
        Args:
            position: Current agent position (x, y)
            max_range: Maximum sensor range
            
        Returns:
            List of 8 distances (one for each direction), normalized to [0, 1]
            where 0 = obstacle at current position, 1 = no obstacle within max_range
        """
        x, y = position
        distances = []
        
        # 8 directions matching our action space (action 0-7)
        # Reordered to match action directions exactly
        directions = [
            (1, 0),     # Action 0: East
            (1, 1),     # Action 1: Southeast  
            (0, 1),     # Action 2: South
            (-1, 1),    # Action 3: Southwest
            (-1, 0),    # Action 4: West
            (-1, -1),   # Action 5: Northwest
            (0, -1),    # Action 6: North
            (1, -1)     # Action 7: Northeast
        ]
        
        for dx, dy in directions:
            # Normalize direction vector
            length = math.sqrt(dx*dx + dy*dy)
            dx_norm, dy_norm = dx/length, dy/length
            
            # Cast ray in this direction
            distance = self._cast_ray(x, y, dx_norm, dy_norm, max_range)
            
            # Normalize distance to [0, 1] where 1 = max_range, 0 = at current position
            normalized_distance = min(distance / max_range, 1.0)
            distances.append(normalized_distance)
        
        return distances
    
    def _cast_ray(self, start_x: float, start_y: float, dx: float, dy: float, max_range: float) -> float:
        """Cast a ray from start position in direction (dx, dy) until hitting obstacle.
        
        Args:
            start_x, start_y: Starting position
            dx, dy: Normalized direction vector
            max_range: Maximum distance to check
            
        Returns:
            Distance to nearest obstacle
        """
        step = 0.05  # Small step size for ray casting
        current_x, current_y = start_x, start_y
        distance = 0.0
        
        while distance < max_range:
            # Move along ray
            current_x += dx * step
            current_y += dy * step
            distance += step
            
            # Check bounds (treat as obstacles)
            if (current_x < 0 or current_y < 0 or 
                current_x >= self.grid.shape[0] or current_y >= self.grid.shape[1]):
                return distance
            
            # Check grid obstacles
            grid_x, grid_y = int(current_x), int(current_y)
            if self.grid[grid_x, grid_y] == 1 or self.grid[grid_x, grid_y] == 2:  # Wall or obstacle
                return distance
            
            # Check for target (also counts as obstacle for sensor)
            if self.grid[grid_x, grid_y] == 3:  # Target
                return distance
            
            # Check dynamic objects
            for obj in self.objects:
                if obj.collides_with(current_x, current_y, 0.0):
                    return distance
        
        return max_range

    def get_enhanced_state(self) -> tuple:
        """Get enhanced state including position and distance sensors.
        
        Returns:
            Tuple with (x, y, sensor_0, sensor_1, ..., sensor_7)
            where sensors are distances to obstacles in 8 directions
        """
        x, y = self.agent_pos
        sensors = self.get_distance_sensors(self.agent_pos)
        return (x, y, *sensors)  # Unpack sensors into tuple
