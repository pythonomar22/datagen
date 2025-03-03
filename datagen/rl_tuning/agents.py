"""
RL agents for data generation optimization.

This module provides reinforcement learning agent implementations for
optimizing synthetic data generation, including policy gradient methods
like REINFORCE and random search.
"""

import logging
import random
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for RL agents."""
    
    def __init__(self, config):
        """
        Initialize the agent.
        
        Args:
            config: Configuration object with RL settings
        """
        self.config = config
        self.state = {}
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action based on the current state.
        
        Args:
            state: Dictionary representing the current state
            
        Returns:
            Dictionary representing the selected action
        """
        raise NotImplementedError
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the agent's policy based on feedback.
        
        Args:
            state: Dictionary representing the state
            action: Dictionary representing the action taken
            reward: Reward received
            next_state: Dictionary representing the next state
            done: Whether the episode is done
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            path: Path to save the state to
        """
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: Path to load the state from
        """
        raise NotImplementedError


class RandomSearchAgent(BaseAgent):
    """
    Simple random search agent that randomly samples parameters
    and retains the best performing ones.
    """
    
    def __init__(self, config):
        """
        Initialize the random search agent.
        
        Args:
            config: Configuration object with RL settings
        """
        super().__init__(config)
        self.best_params = None
        self.best_reward = -float('inf')
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action by randomly sampling parameters.
        
        Args:
            state: Dictionary with current state (unused in random search)
            
        Returns:
            Dictionary with sampled parameters
        """
        action = {}
        
        # Sample temperature
        action['temperature'] = round(
            random.uniform(
                self.config.rl_tuning.min_temperature,
                self.config.rl_tuning.max_temperature
            ), 
            2
        )
        
        # Sample top_p
        action['top_p'] = round(
            random.uniform(
                self.config.rl_tuning.min_top_p,
                self.config.rl_tuning.max_top_p
            ), 
            2
        )
        
        # Sample generation method if multiple methods are enabled
        if len(self.config.rl_tuning.generation_methods) > 1:
            action['generation_method'] = random.choice(
                self.config.rl_tuning.generation_methods
            )
        else:
            action['generation_method'] = self.config.rl_tuning.generation_methods[0]
        
        return action
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update the best parameters if the reward is improved.
        
        Args:
            state: Dictionary with state (unused)
            action: Dictionary with the action parameters
            reward: Reward received for the action
            next_state: Dictionary with next state (unused)
            done: Whether the episode is done (unused)
        """
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = action.copy()
            logger.info(f"Random search found new best parameters: {self.best_params}")
            logger.info(f"New best reward: {self.best_reward}")
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            path: Path to save the state to
        """
        state = {
            'best_params': self.best_params,
            'best_reward': self.best_reward
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved RandomSearchAgent state to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: Path to load the state from
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.best_params = state.get('best_params')
        self.best_reward = state.get('best_reward', -float('inf'))
        
        logger.info(f"Loaded RandomSearchAgent state from {path}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best reward: {self.best_reward}")


class PolicyNetwork(nn.Module):
    """Neural network policy for REINFORCE algorithm."""
    
    def __init__(self, state_dim: int, continuous_action_dim: int, 
                 discrete_action_dims: List[int], hidden_dim: int = 64):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state vector
            continuous_action_dim: Number of continuous actions
            discrete_action_dims: List of dimensions for each discrete action
            hidden_dim: Size of hidden layer
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PolicyNetwork but not available")
        
        self.state_dim = state_dim
        self.continuous_action_dim = continuous_action_dim
        self.discrete_action_dims = discrete_action_dims
        
        # Shared feature extraction layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Output layers for continuous actions (mean and log_std)
        if continuous_action_dim > 0:
            self.mean_head = nn.Linear(hidden_dim, continuous_action_dim)
            self.log_std_head = nn.Linear(hidden_dim, continuous_action_dim)
        
        # Output layers for discrete actions
        self.categorical_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in discrete_action_dims
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of distribution parameters for actions
        """
        x = F.relu(self.fc1(state))
        
        outputs = []
        
        # Continuous action distributions
        if self.continuous_action_dim > 0:
            means = self.mean_head(x)
            log_stds = self.log_std_head(x)
            log_stds = torch.clamp(log_stds, -20, 2)  # Prevent numerical instability
            outputs.append((means, log_stds))
        
        # Discrete action distributions
        for head in self.categorical_heads:
            logits = head(x)
            outputs.append(logits)
        
        return tuple(outputs)


class REINFORCEAgent(BaseAgent):
    """
    Agent implementing the REINFORCE algorithm (policy gradient).
    """
    
    def __init__(self, config):
        """
        Initialize the REINFORCE agent.
        
        Args:
            config: Configuration object with RL settings
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for REINFORCEAgent but not available")
        
        # Define state and action dimensions
        # State: previous reward, continuous parameters (temperature, top_p)
        self.state_dim = 3  # previous_reward, temperature, top_p
        
        # Actions: continuous (temperature, top_p) and discrete (generation_method)
        self.continuous_action_dim = 2  # temperature, top_p
        self.discrete_action_dims = [len(self.config.rl_tuning.generation_methods)]
        
        # Create policy network
        self.policy = PolicyNetwork(
            state_dim=self.state_dim,
            continuous_action_dim=self.continuous_action_dim,
            discrete_action_dims=self.discrete_action_dims,
            hidden_dim=self.config.rl_tuning.policy_hidden_dim
        )
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.config.rl_tuning.learning_rate
        )
        
        # Initialize trajectory storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        
        # Best parameters tracking
        self.best_params = None
        self.best_reward = -float('inf')
        
        logger.info("Initialized REINFORCE agent")
    
    def _normalize_state(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dictionary to normalized tensor.
        
        Args:
            state: Dictionary with state variables
            
        Returns:
            Normalized state tensor
        """
        # Extract state variables
        prev_reward = state.get('previous_reward', 0.0)
        temp = state.get('temperature', 0.7)
        top_p = state.get('top_p', 0.9)
        
        # Normalize values to reasonable ranges
        norm_reward = np.clip(prev_reward, -1.0, 1.0)  # Assume reward is normalized already
        norm_temp = (temp - self.config.rl_tuning.min_temperature) / (
            self.config.rl_tuning.max_temperature - self.config.rl_tuning.min_temperature
        )
        norm_top_p = (top_p - self.config.rl_tuning.min_top_p) / (
            self.config.rl_tuning.max_top_p - self.config.rl_tuning.min_top_p
        )
        
        # Create tensor
        return torch.tensor([norm_reward, norm_temp, norm_top_p], dtype=torch.float32)
    
    def _denormalize_continuous_action(self, action_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Convert normalized continuous action tensor to actual parameter values.
        
        Args:
            action_tensor: Tensor with normalized continuous actions
            
        Returns:
            Tuple of (temperature, top_p)
        """
        # Rescale actions to parameter ranges
        temperature = (
            action_tensor[0].item() * 
            (self.config.rl_tuning.max_temperature - self.config.rl_tuning.min_temperature) + 
            self.config.rl_tuning.min_temperature
        )
        
        top_p = (
            action_tensor[1].item() * 
            (self.config.rl_tuning.max_top_p - self.config.rl_tuning.min_top_p) + 
            self.config.rl_tuning.min_top_p
        )
        
        # Round and clip values
        temperature = round(np.clip(temperature, 
                                    self.config.rl_tuning.min_temperature,
                                    self.config.rl_tuning.max_temperature), 2)
        
        top_p = round(np.clip(top_p,
                              self.config.rl_tuning.min_top_p,
                              self.config.rl_tuning.max_top_p), 2)
        
        return temperature, top_p
    
    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action using the policy network.
        
        Args:
            state: Dictionary with current state
            
        Returns:
            Dictionary with selected action parameters
        """
        # Convert state to tensor
        state_tensor = self._normalize_state(state)
        
        # Forward pass through policy network
        with torch.no_grad():
            policy_output = self.policy(state_tensor)
        
        # Extract continuous action distribution parameters
        means, log_stds = policy_output[0]
        stds = torch.exp(log_stds)
        
        # Create normal distributions for continuous actions
        normal_dists = Normal(means, stds)
        
        # Sample continuous actions
        continuous_actions = normal_dists.sample()
        continuous_log_probs = normal_dists.log_prob(continuous_actions).sum()
        
        # Extract discrete action logits
        discrete_logits = policy_output[1]
        
        # Create categorical distribution for discrete action
        categorical_dist = Categorical(logits=discrete_logits)
        
        # Sample discrete action
        discrete_action = categorical_dist.sample()
        discrete_log_prob = categorical_dist.log_prob(discrete_action)
        
        # Calculate total log probability
        total_log_prob = continuous_log_probs + discrete_log_prob
        
        # Convert actions to actual parameter values
        temperature, top_p = self._denormalize_continuous_action(continuous_actions)
        generation_method = self.config.rl_tuning.generation_methods[discrete_action.item()]
        
        # Create action dictionary
        action = {
            'temperature': temperature,
            'top_p': top_p,
            'generation_method': generation_method
        }
        
        # Store trajectory data
        self.states.append(state_tensor)
        self.actions.append((continuous_actions, discrete_action))
        self.log_probs.append(total_log_prob)
        
        return action
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], 
               reward: float, next_state: Dict[str, Any], done: bool) -> None:
        """
        Update policy using the REINFORCE algorithm.
        
        Args:
            state: Dictionary with state
            action: Dictionary with action parameters
            reward: Reward received
            next_state: Dictionary with next state
            done: Whether the episode is done
        """
        # Store reward
        self.rewards.append(reward)
        
        # Track best parameters
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = action.copy()
            logger.info(f"REINFORCE found new best parameters: {self.best_params}")
            logger.info(f"New best reward: {self.best_reward}")
        
        # Only update policy at the end of an episode
        if done:
            # Convert rewards to returns (cumulative future rewards)
            returns = []
            R = 0
            for r in reversed(self.rewards):
                R = r + self.config.rl_tuning.gamma * R
                returns.insert(0, R)
            
            # Normalize returns
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Calculate loss
            policy_loss = []
            for log_prob, R in zip(self.log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Clear trajectory data
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            
            logger.info(f"Updated policy with loss: {policy_loss.item()}")
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        
        Args:
            path: Path to save the state to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_path = path + '.model'
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_params': self.best_params,
            'best_reward': self.best_reward
        }, model_path)
        
        logger.info(f"Saved REINFORCE agent state to {model_path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        
        Args:
            path: Path to load the state from
        """
        # Load model state
        model_path = path + '.model'
        checkpoint = torch.load(model_path)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_params = checkpoint['best_params']
        self.best_reward = checkpoint['best_reward']
        
        logger.info(f"Loaded REINFORCE agent state from {model_path}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best reward: {self.best_reward}") 