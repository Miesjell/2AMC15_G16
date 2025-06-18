"""
Ultimate PPO Agent - Adapted to work with simple ContinuousEnvironment and train.py

This PPO implementation is designed to work with the existing train.py structure:
1. Works with the original ContinuousEnvironment (no modifications needed)
2. Handles episode management internally within the agent
3. Compatible with the simple training loop in train.py
4. Incorporates all the critical PPO fixes for optimal performance
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from agents.base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Policy network that outputs action logits."""
    
    def __init__(self, state_dim: int = 6, action_dim: int = 4, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()
        
        # Compact network for grid environments
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Proper initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for final layer
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
    
    def forward(self, state):
        """Forward pass returning action logits."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        return action_logits


class ValueNetwork(nn.Module):
    """Value network for state value estimation."""
    
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        """Forward pass returning state value."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_head(x)
        return value


class PPOAgent(BaseAgent):
    """PPO agent adapted for simple train.py interface."""
    
    def __init__(self, env,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 hidden_dim: int = 64,
                 batch_size: int = 64,
                 update_frequency: int = 128,  # Update after this many steps
                 entropy_coef: float = 0.02,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5):
        super().__init__()
        
        # Environment reference
        self.env = env
        self.state_dim = 6
        self.action_dim = 4
        
        # Training hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Device configuration with better CUDA detection
        self.device = self._get_device()
        print(f"Ultimate PPO Agent using device: {self.device}")
        
        # Networks
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim, 
            action_dim=self.action_dim, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim=self.state_dim, 
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1e-5)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr, eps=1e-5)
        
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
        
        # Training state
        self.training = True
        self.episode_count = 0
        self.total_steps = 0
        self.steps_since_update = 0
        
        # Store previous state and action for experience collection
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        
        # Performance tracking
        self.recent_losses = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        
    def take_action(self, state: np.ndarray) -> int:
        """Select action using policy network."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.policy_net(state_tensor)
            
            if self.training:
                # Sample from policy distribution for exploration
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                action = action.item()
                log_prob = log_prob.item()
            else:
                # Greedy action for evaluation
                action = torch.argmax(action_logits).item()
                log_prob = 0.0
        
        # Store for experience collection
        self.prev_state = state.copy()
        self.prev_action = action
        self.prev_log_prob = log_prob
        
        return action
    
    def update(self, state: np.ndarray, reward: float, actual_action: int):
        """Update agent with transition experience."""
        if not self.training or self.prev_state is None:
            return
        
        # Convert states to tensors
        prev_state_tensor = torch.tensor(self.prev_state, dtype=torch.float32).to(self.device)
        current_state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Get value estimates
        with torch.no_grad():
            prev_value = self.value_net(prev_state_tensor.unsqueeze(0)).item()
        
        # Store transition in buffer
        self.memory['states'].append(prev_state_tensor.cpu())
        self.memory['actions'].append(self.prev_action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(current_state_tensor.cpu())
        self.memory['dones'].append(False)  # Will be set properly in reset()
        self.memory['log_probs'].append(self.prev_log_prob)
        self.memory['values'].append(prev_value)
        
        self.total_steps += 1
        self.steps_since_update += 1
        self.recent_rewards.append(reward)
        
        # Train periodically based on step count
        if self.steps_since_update >= self.update_frequency and len(self.memory['states']) >= self.batch_size:
            loss = self._train()
            if loss is not None:
                self.recent_losses.append(loss)
            self._clear_memory()
            self.steps_since_update = 0
            return loss
    
    def reset(self):
        """Reset agent for new episode."""
        # Mark last transition as terminal if we have data
        if len(self.memory['states']) > 0:
            self.memory['dones'][-1] = True
        
        # Train if we have accumulated experience
        if len(self.memory['states']) >= self.batch_size:
            loss = self._train()
            if loss is not None:
                self.recent_losses.append(loss)
            self._clear_memory()
            self.steps_since_update = 0
        
        # Reset episode state
        self.prev_state = None
        self.prev_action = None
        self.prev_log_prob = None
        self.episode_count += 1
    
    def _train(self):
        """Train the agent using PPO algorithm."""
        if len(self.memory['states']) == 0:
            return None
        
        # Convert experience to tensors
        states = torch.stack(self.memory['states']).to(self.device)
        actions = torch.tensor(self.memory['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(self.memory['rewards'], dtype=torch.float32).to(self.device)
        next_states = torch.stack(self.memory['next_states']).to(self.device)
        dones = torch.tensor(self.memory['dones'], dtype=torch.bool).to(self.device)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory['values'], dtype=torch.float32).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            next_values = self.value_net(next_states).squeeze()
            advantages, returns = self._compute_gae(rewards, old_values, next_values, dones)
            
            # Normalize advantages for stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_loss = 0.0
        n_updates = 0
        
        for _ in range(self.k_epochs):
            # Randomize batch order
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                action_logits = self.policy_net(batch_states)
                values = self.value_net(batch_states).squeeze()
                
                # Policy loss
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.eps_clip, self.eps_clip
                )
                value_loss1 = F.mse_loss(values, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                total_loss += loss.item()
                n_updates += 1
        
        return total_loss / max(1, n_updates)
    
    def _compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # Terminal state handling
            next_non_terminal = ~dones[t]
            next_value = next_values[t]
            
            # GAE computation
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def _clear_memory(self):
        """Clear experience buffer."""
        for key in self.memory:
            self.memory[key].clear()
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'avg_recent_loss': np.mean(self.recent_losses) if self.recent_losses else 0.0,
            'avg_recent_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
            'value_lr': self.value_optimizer.param_groups[0]['lr'],
            'buffer_size': len(self.memory['states'])
        }
    
    def save_model(self, filepath: str):
        """Save trained model."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training = training
        if training:
            self.policy_net.train()
            self.value_net.train()
        else:
            self.policy_net.eval()
            self.value_net.eval()
    
    def _get_device(self):
        """Get the best available device with robust CUDA detection."""
        try:
            if torch.cuda.is_available():
                # Test if CUDA actually works by creating a small tensor
                test_tensor = torch.tensor([1.0], device='cuda')
                test_result = test_tensor + 1.0
                # If we get here, CUDA works
                cuda_device = torch.device('cuda')
                print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                return cuda_device
        except Exception as e:
            print(f"CUDA available but failed to initialize: {e}")
            print("Falling back to CPU")
        
        return torch.device('cpu')
