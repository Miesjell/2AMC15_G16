"""
PPO Agent Implementation.

This implements Proximal Policy Optimization (PPO) for continuous state spaces
with discrete actions (8 directions).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from agents.continuous_base_agent import ContinuousBaseAgent


class PolicyNetwork(nn.Module):
    """Policy network for continuous state spaces with discrete actions."""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 8, hidden_dim: int = 256):  # Changed from 2 to 10
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state, temperature: float = 1.0):
        """Forward pass through the policy network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #scale logits instead of probabilites by getting logits first and dividing by temperature:
        logits = self.action_head(x)
        scaled_logits = logits / temperature  
        return F.softmax(scaled_logits, dim=-1)


class ValueNetwork(nn.Module):
    """Value network for continuous state spaces."""
    
    def __init__(self, state_dim: int = 10, hidden_dim: int = 128):  # Changed from 2 to 10
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """Forward pass through the value network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_head(x)
        return value


class PPOAgent(ContinuousBaseAgent):
    """PPO agent for continuous state spaces with discrete actions."""
    
    def __init__(self, state_bounds: tuple = ((0.0, 10.0), (0.0, 10.0)), 
                 action_dim: int = 8,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 hidden_dim: int = 128,
                 batch_size: int = 64,
                 buffer_size: int = 2048,
                 epsilon: float = 0.1,  # Reduced initial exploration
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995,  # Slower decay
                 temperature: float = 1.2):  # Reduced temperature
        super().__init__()
        
        # Environment parameters
        self.state_bounds = state_bounds
        self.action_dim = action_dim
        
        # Training parameters
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim=10, action_dim=action_dim, 
                                                hidden_dim=hidden_dim)
        self.value_net = ValueNetwork(state_dim=10, hidden_dim=hidden_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        # Training mode flag
        self.training = True
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.update_counter = 0  # Counter for training frequency
        
    def normalize_state(self, state: tuple) -> torch.Tensor:
        """Normalize enhanced state (position + sensors) to appropriate ranges.
        
        Args:
            state: Tuple with (x, y, sensor_0, sensor_1, ..., sensor_7)
            
        Returns:
            Normalized tensor of shape (10,)
        """
        x, y = state[0], state[1]  # Position
        sensors = state[2:]  # 8 sensor readings
        
        # Normalize position to [-1, 1] range
        x_min, x_max = self.state_bounds[0]
        y_min, y_max = self.state_bounds[1]
        
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
        
        # Clamp position to bounds
        x_norm = max(-1.0, min(1.0, x_norm))
        y_norm = max(-1.0, min(1.0, y_norm))
        
        # Sensors are already normalized to [0, 1], convert to [-1, 1]
        sensors_norm = [2 * s - 1 for s in sensors]
        
        # Combine normalized position and sensors
        normalized_state = [x_norm, y_norm] + sensors_norm
        
        return torch.tensor(normalized_state, dtype=torch.float32)
    
    def take_action(self, state: tuple) -> int:
        """Select an action given the current enhanced state."""
        state_tensor = self.normalize_state(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor, self.temperature)
            
            if self.training:
                # Add exploration noise and epsilon-greedy
                if np.random.random() < self.epsilon:
                    # Random exploration
                    action = np.random.randint(0, self.action_dim)
                    # Calculate log prob for the random action
                    dist = torch.distributions.Categorical(action_probs)
                    self._temp_log_prob = dist.log_prob(torch.tensor(action)).item()
                else:
                    
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
                    self._temp_log_prob = dist.log_prob(torch.tensor(action)).item()
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                return action
            else:
                # Take best action during evaluation
                return torch.argmax(action_probs).item()
    
    def update(self, state: tuple, reward: float, action: int, 
               next_state: tuple, done: bool):
        """Update the agent with experience."""
        if not self.training:
            return
        
        # Store experience
        state_tensor = self.normalize_state(state)
        next_state_tensor = self.normalize_state(next_state)
        
        # Get value estimate for current state
        with torch.no_grad():
            value = self.value_net(state_tensor.unsqueeze(0)).item()
        
        self.memory['states'].append(state_tensor)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state_tensor)
        self.memory['dones'].append(done)
        self.memory['log_probs'].append(self._temp_log_prob)
        self.memory['values'].append(value)
        
        self.total_steps += 1
        self.update_counter += 1
        
        # Train less frequently - every 50 steps or when buffer is full
        should_train = (self.update_counter >= 50 and len(self.memory['states']) >= self.batch_size) or \
                      len(self.memory['states']) >= self.buffer_size or \
                      (done and len(self.memory['states']) >= self.batch_size)
        
        if should_train:
            loss = self._train()
            self._clear_memory()
            self.update_counter = 0
            return loss
            
        if done:
            self.episode_count += 1
    
    def _train(self):
        """Train the PPO agent."""
        if len(self.memory['states']) == 0:
            return 0.0
        
        # Convert lists to tensors
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'], dtype=torch.long)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32)
        next_states = torch.stack(self.memory['next_states'])
        dones = torch.tensor(self.memory['dones'], dtype=torch.bool)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32)
        old_values = torch.tensor(self.memory['values'], dtype=torch.float32)
        
        # Calculate advantages and returns
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            returns = self._calculate_returns(rewards, next_values, dones)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # PPO training loop
        for _ in range(self.k_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Policy update
                action_probs = self.policy_net(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                # Value update
                new_values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Optimize policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Optimize value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        return (total_policy_loss + total_value_loss) / (self.k_epochs * max(1, len(states) // self.batch_size))
    
    def _calculate_returns(self, rewards: torch.Tensor, next_values: torch.Tensor, 
                          dones: torch.Tensor) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            else:
                running_return = rewards[t] + self.gamma * running_return
            if t < len(returns) - 1:
                running_return += self.gamma * next_values[t] * (1 - dones[t].float())
            returns[t] = running_return
        
        return returns
    
    def _clear_memory(self):
        """Clear the experience buffer."""
        for key in self.memory:
            self.memory[key].clear()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training = training
        if training:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.policy_net.eval()
            self.value_net.eval()
