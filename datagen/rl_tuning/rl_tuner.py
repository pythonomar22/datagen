"""
RL-guided data generation tuner.

This module provides the RLTuner class, which uses reinforcement learning
to optimize the generation of synthetic data for a specific target model
and task.
"""

import os
import json
import logging
import random
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple, Union
import copy
import time

from datagen import Generator, Config, Results
from .agents import RandomSearchAgent, REINFORCEAgent, TORCH_AVAILABLE

logger = logging.getLogger(__name__)


class RLTuner:
    """
    Reinforcement Learning-guided data generation optimizer.
    
    This class implements a reinforcement learning loop that optimizes
    the generation of synthetic data for improving a user's model on a
    specific task.
    """
    
    def __init__(
        self, 
        config: Config,
        target_model: Callable[[List[Dict[str, Any]]], Dict[str, float]],
        validation_dataset: List[Dict[str, Any]], 
        generator: Optional[Generator] = None
    ):
        """
        Initialize the RLTuner.
        
        Args:
            config: Config object with RL-specific settings
            target_model: Function that takes a list of dictionaries (synthetic examples)
                          and returns a dictionary of evaluation metrics
            validation_dataset: List of dictionaries (validation examples) for evaluation
            generator: Optional Generator instance. If None, a new one will be created.
        """
        self.config = config
        self.target_model = target_model
        self.validation_dataset = validation_dataset
        
        # Create generator if not provided
        if generator is None:
            self.generator = Generator(config)
        else:
            self.generator = generator
        
        # Initialize state tracking
        self.best_reward = -float('inf')
        self.best_params = {
            'temperature': self.config.sampling.temperature,
            'top_p': self.config.sampling.top_p,
            'generation_method': 'self_instruct' if self.config.generation.self_instruct else 'evol_instruct'
        }
        
        self.current_params = copy.deepcopy(self.best_params)
        self.history = []
        
        # Initialize the appropriate RL agent based on configuration
        self._init_agent()
        
        logger.info(f"Initialized RLTuner with algorithm: {self.config.rl_tuning.rl_algorithm}")
        logger.info(f"Target reward metric: {self.config.rl_tuning.reward_metric}")
        logger.info(f"Initial parameters: {self.current_params}")
    
    def _init_agent(self):
        """Initialize the appropriate RL agent based on configuration."""
        if self.config.rl_tuning.rl_algorithm == "random_search":
            self.agent = RandomSearchAgent(self.config)
            logger.info("Using RandomSearchAgent for parameter optimization")
        elif self.config.rl_tuning.rl_algorithm == "reinforce":
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available. Falling back to random search.")
                self.agent = RandomSearchAgent(self.config)
                self.config.rl_tuning.rl_algorithm = "random_search"
            else:
                # Check if required config parameters are present
                if not hasattr(self.config.rl_tuning, "policy_hidden_dim"):
                    self.config.rl_tuning.policy_hidden_dim = 64
                if not hasattr(self.config.rl_tuning, "gamma"):
                    self.config.rl_tuning.gamma = 0.99
                
                self.agent = REINFORCEAgent(self.config)
                logger.info("Using REINFORCEAgent (policy gradient) for parameter optimization")
        else:
            logger.warning(f"Unknown RL algorithm: {self.config.rl_tuning.rl_algorithm}. Using random search.")
            self.agent = RandomSearchAgent(self.config)
            self.config.rl_tuning.rl_algorithm = "random_search"
    
    def train(
        self, 
        num_iterations: Optional[int] = None, 
        batch_size: Optional[int] = None,
        seed_examples: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Run the RL training loop to optimize data generation.
        
        Args:
            num_iterations: Number of RL iterations. If None, use config value.
            batch_size: Number of examples to generate per iteration. If None, use config value.
            seed_examples: Optional initial seed examples for generation.
                          If None, the generator's default or random seeds will be used.
                          
        Returns:
            Dictionary with training results and history.
        """
        # Use config values if not specified
        num_iterations = num_iterations or self.config.rl_tuning.num_iterations
        batch_size = batch_size or self.config.rl_tuning.batch_size
        
        logger.info(f"Starting RL training for {num_iterations} iterations with batch size {batch_size}")
        
        # Calculate baseline performance (optional)
        baseline_metrics = None
        if seed_examples:
            logger.info("Calculating baseline performance with provided seed examples")
            baseline_metrics = self.target_model(seed_examples)
            initial_reward = baseline_metrics.get(self.config.rl_tuning.reward_metric, 0)
            logger.info(f"Baseline {self.config.rl_tuning.reward_metric}: {initial_reward}")
            if initial_reward > self.best_reward:
                self.best_reward = initial_reward
        
        # Initial state
        state = {
            'previous_reward': self.best_reward,
            'temperature': self.current_params['temperature'],
            'top_p': self.current_params['top_p'],
        }
        
        # Main RL loop
        for iteration in range(num_iterations):
            logger.info(f"Starting iteration {iteration+1}/{num_iterations}")
            iteration_start_time = time.time()
            
            # 1. Select action using the RL agent
            action = self.agent.select_action(state)
            logger.info(f"Selected parameters: {action}")
            
            # 2. Update current parameters and generator settings
            self.current_params = copy.deepcopy(action)
            self._update_generator_settings()
            
            # 3. Generate synthetic data
            if self.current_params['generation_method'] == 'self_instruct':
                generation_results = self.generator.generate_from_seed(
                    seed_examples=seed_examples, 
                    count=batch_size,
                    method="self_instruct"
                )
            elif self.current_params['generation_method'] == 'evol_instruct':
                # For simplicity in MVP, if using evol_instruct, extract instructions from seed examples
                if seed_examples:
                    initial_instructions = [ex["instruction"] for ex in seed_examples if "instruction" in ex]
                else:
                    # Default simple instructions if no seed examples are provided
                    initial_instructions = [
                        "Explain how a refrigerator works.",
                        "Write a short poem about technology.",
                        "Describe the water cycle."
                    ]
                generation_results = self.generator.evolve_instructions(
                    instructions=initial_instructions,
                    rounds=self.config.generation.evol_rounds
                )
            else:
                raise ValueError(f"Unknown generation method: {self.current_params['generation_method']}")
            
            logger.info(f"Generated {len(generation_results)} examples")
            
            # 4. Evaluate on target model
            logger.info("Evaluating generated data on target model")
            metrics = self.target_model(generation_results.data)
            reward = metrics.get(self.config.rl_tuning.reward_metric, 0)
            logger.info(f"Evaluation result - {self.config.rl_tuning.reward_metric}: {reward}")
            
            # 5. Update best parameters if improved
            if reward > self.best_reward:
                logger.info(f"New best reward: {reward} (previous: {self.best_reward})")
                self.best_reward = reward
                self.best_params = copy.deepcopy(self.current_params)
            else:
                logger.info(f"No improvement. Best reward remains: {self.best_reward}")
            
            # 6. Calculate next state
            next_state = {
                'previous_reward': reward,
                'temperature': self.current_params['temperature'],
                'top_p': self.current_params['top_p'],
            }
            
            # 7. Update the RL agent
            is_last_iteration = (iteration == num_iterations - 1)
            self.agent.update(state, action, reward, next_state, done=is_last_iteration)
            
            # 8. Prepare for next iteration
            state = next_state
            
            # 9. Record history
            iteration_time = time.time() - iteration_start_time
            history_entry = {
                'iteration': iteration + 1,
                'params': copy.deepcopy(self.current_params),
                'metrics': metrics,
                'reward': reward,
                'examples_count': len(generation_results),
                'time_seconds': iteration_time
            }
            self.history.append(history_entry)
            
            logger.info(f"Completed iteration {iteration+1} in {iteration_time:.2f} seconds")
        
        # Return results
        results = {
            'best_params': self.best_params,
            'best_reward': self.best_reward,
            'baseline_metrics': baseline_metrics,
            'history': self.history,
            'total_iterations': num_iterations,
            'reward_metric': self.config.rl_tuning.reward_metric,
            'algorithm': self.config.rl_tuning.rl_algorithm
        }
        
        # Log final results
        logger.info("RL training completed")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best {self.config.rl_tuning.reward_metric}: {self.best_reward}")
        
        return results
    
    def _sample_parameters_random(self) -> None:
        """
        Sample parameters randomly within configured ranges.
        Updates self.current_params in place.
        
        Note: This method is kept for backward compatibility but is deprecated.
        Use the RandomSearchAgent instead.
        """
        # Sample temperature
        self.current_params['temperature'] = round(
            random.uniform(
                self.config.rl_tuning.min_temperature,
                self.config.rl_tuning.max_temperature
            ), 
            2
        )
        
        # Sample top_p
        self.current_params['top_p'] = round(
            random.uniform(
                self.config.rl_tuning.min_top_p,
                self.config.rl_tuning.max_top_p
            ), 
            2
        )
        
        # Sample generation method if multiple methods are enabled
        if len(self.config.rl_tuning.generation_methods) > 1:
            self.current_params['generation_method'] = random.choice(
                self.config.rl_tuning.generation_methods
            )
    
    def _update_generator_settings(self) -> None:
        """
        Update the generator settings based on current parameters.
        """
        # Update sampling parameters
        self.generator.config.sampling.temperature = self.current_params['temperature']
        self.generator.config.sampling.top_p = self.current_params['top_p']
        
        # Update generation method
        if self.current_params['generation_method'] == 'self_instruct':
            self.generator.config.generation.self_instruct = True
            self.generator.config.generation.evol_instruct = False
        elif self.current_params['generation_method'] == 'evol_instruct':
            self.generator.config.generation.self_instruct = False
            self.generator.config.generation.evol_instruct = True
    
    def generate_with_best_params(
        self, 
        count: int, 
        seed_examples: Optional[List[Dict[str, str]]] = None
    ) -> Results:
        """
        Generate synthetic data using the best parameters found during training.
        
        Args:
            count: Number of examples to generate
            seed_examples: Optional seed examples for generation
            
        Returns:
            Results object containing the generated examples
        """
        # Store current parameters
        temp_params = copy.deepcopy(self.current_params)
        
        # Set best parameters
        self.current_params = copy.deepcopy(self.best_params)
        self._update_generator_settings()
        
        # Generate with best parameters
        logger.info(f"Generating {count} examples with best parameters: {self.best_params}")
        
        if self.best_params['generation_method'] == 'self_instruct':
            results = self.generator.generate_from_seed(
                seed_examples=seed_examples, 
                count=count,
                method="self_instruct"
            )
        elif self.best_params['generation_method'] == 'evol_instruct':
            if seed_examples:
                initial_instructions = [ex["instruction"] for ex in seed_examples if "instruction" in ex]
            else:
                initial_instructions = [
                    "Explain how a refrigerator works.",
                    "Write a short poem about technology.",
                    "Describe the water cycle."
                ]
            
            # Need to estimate how many instructions to evolve to get approximately 'count' examples
            # This is approximate since evolution can expand instructions
            approx_instruction_count = max(3, int(count / 3))
            seed_subset = initial_instructions[:min(len(initial_instructions), approx_instruction_count)]
            
            results = self.generator.evolve_instructions(
                instructions=seed_subset,
                rounds=self.config.generation.evol_rounds
            )
        
        # Restore original parameters
        self.current_params = temp_params
        self._update_generator_settings()
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save the RLTuner state to a file.
        
        Args:
            path: Path to save the state to
        """
        # First save basic state
        state = {
            'best_params': self.best_params,
            'best_reward': self.best_reward,
            'history': self.history,
            'config': {
                'sampling': self.config.sampling.__dict__,
                'rl_tuning': self.config.rl_tuning.__dict__
            },
            'algorithm': self.config.rl_tuning.rl_algorithm
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Then save agent-specific state
        agent_path = path + '.agent'
        self.agent.save(agent_path)
        
        logger.info(f"Saved RLTuner state to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the RLTuner state from a file.
        
        Args:
            path: Path to load the state from
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.best_params = state['best_params']
        self.best_reward = state['best_reward']
        self.history = state['history']
        
        # Update config if present
        if 'config' in state:
            if 'sampling' in state['config']:
                for key, value in state['config']['sampling'].items():
                    setattr(self.config.sampling, key, value)
            
            if 'rl_tuning' in state['config']:
                for key, value in state['config']['rl_tuning'].items():
                    setattr(self.config.rl_tuning, key, value)
        
        # Check for algorithm change
        if 'algorithm' in state:
            self.config.rl_tuning.rl_algorithm = state['algorithm']
        
        # Reinitialize agent based on loaded config
        self._init_agent()
        
        # Load agent state
        agent_path = path + '.agent'
        if os.path.exists(agent_path):
            self.agent.load(agent_path)
        
        logger.info(f"Loaded RLTuner state from {path}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best reward: {self.best_reward}") 