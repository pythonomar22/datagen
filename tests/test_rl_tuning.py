"""
Tests for the RL-guided data generation module.
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.rl_tuning import RLTuner


class TestRLTuner(unittest.TestCase):
    """Tests for the RLTuner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.config.rl_tuning.enable_rl_tuning = True
        self.config.rl_tuning.num_iterations = 2  # Small for testing
        self.config.rl_tuning.batch_size = 3  # Small for testing
        
        # Mock validation dataset
        self.validation_dataset = [
            {"text": "This is positive", "label": 1},
            {"text": "This is negative", "label": 0}
        ]
        
        # Mock target model function
        self.target_model = Mock()
        self.target_model.return_value = {"accuracy": 0.8, "f1": 0.75}
        
        # Mock generator
        self.generator = MagicMock()
        self.generator.config = self.config
        
        # Mock generate_from_seed method
        mock_results = Results([
            {"instruction": "Test", "response": "Response 1"},
            {"instruction": "Test", "response": "Response 2"},
            {"instruction": "Test", "response": "Response 3"}
        ])
        self.generator.generate_from_seed.return_value = mock_results
        
        # Mock evolve_instructions method
        self.generator.evolve_instructions.return_value = mock_results
        
        # Create RLTuner instance
        self.tuner = RLTuner(
            config=self.config,
            target_model=self.target_model,
            validation_dataset=self.validation_dataset,
            generator=self.generator
        )
        
    def test_initialization(self):
        """Test RLTuner initialization."""
        self.assertEqual(self.tuner.config, self.config)
        self.assertEqual(self.tuner.target_model, self.target_model)
        self.assertEqual(self.tuner.validation_dataset, self.validation_dataset)
        self.assertEqual(self.tuner.generator, self.generator)
        
        # Check initial parameters
        self.assertIn('temperature', self.tuner.current_params)
        self.assertIn('top_p', self.tuner.current_params)
        self.assertIn('generation_method', self.tuner.current_params)
        
    def test_sample_parameters_random(self):
        """Test random parameter sampling."""
        # Store original params
        original_params = self.tuner.current_params.copy()
        
        # Sample new params
        self.tuner._sample_parameters_random()
        
        # Verify params changed
        self.assertNotEqual(self.tuner.current_params, original_params)
        
        # Verify params are within bounds
        self.assertGreaterEqual(self.tuner.current_params['temperature'], 
                               self.config.rl_tuning.min_temperature)
        self.assertLessEqual(self.tuner.current_params['temperature'], 
                            self.config.rl_tuning.max_temperature)
        self.assertGreaterEqual(self.tuner.current_params['top_p'], 
                               self.config.rl_tuning.min_top_p)
        self.assertLessEqual(self.tuner.current_params['top_p'], 
                            self.config.rl_tuning.max_top_p)
        
    def test_update_generator_settings(self):
        """Test updating generator settings."""
        # Set test parameters
        self.tuner.current_params = {
            'temperature': 0.5,
            'top_p': 0.8,
            'generation_method': 'self_instruct'
        }
        
        # Update generator settings
        self.tuner._update_generator_settings()
        
        # Verify generator settings updated
        self.assertEqual(self.generator.config.sampling.temperature, 0.5)
        self.assertEqual(self.generator.config.sampling.top_p, 0.8)
        self.assertEqual(self.generator.config.generation.self_instruct, True)
        self.assertEqual(self.generator.config.generation.evol_instruct, False)
        
        # Test evol_instruct method
        self.tuner.current_params['generation_method'] = 'evol_instruct'
        self.tuner._update_generator_settings()
        self.assertEqual(self.generator.config.generation.self_instruct, False)
        self.assertEqual(self.generator.config.generation.evol_instruct, True)
        
    def test_train(self):
        """Test the train method."""
        # Set up mock behavior
        seed_examples = [
            {"instruction": "Test", "response": "This is a test"}
        ]
        
        # Run training
        results = self.tuner.train(seed_examples=seed_examples)
        
        # Verify target model was called with generated examples
        self.target_model.assert_called()
        
        # Verify results structure
        self.assertIn('best_params', results)
        self.assertIn('best_reward', results)
        self.assertIn('history', results)
        
        # Verify history
        self.assertEqual(len(results['history']), self.config.rl_tuning.num_iterations)
        
    def test_generate_with_best_params(self):
        """Test generating with best parameters."""
        # Set best params
        self.tuner.best_params = {
            'temperature': 0.3,
            'top_p': 0.7,
            'generation_method': 'self_instruct'
        }
        
        # Generate with best params
        results = self.tuner.generate_with_best_params(count=5)
        
        # Verify generator was called with correct method
        self.generator.generate_from_seed.assert_called()
        
        # Test with evol_instruct
        self.tuner.best_params['generation_method'] = 'evol_instruct'
        results = self.tuner.generate_with_best_params(count=5)
        self.generator.evolve_instructions.assert_called()
        
    def test_save_load(self):
        """Test saving and loading state."""
        # Set up test state
        self.tuner.best_params = {'temperature': 0.3, 'top_p': 0.7, 'generation_method': 'self_instruct'}
        self.tuner.best_reward = 0.85
        self.tuner.history = [
            {'iteration': 1, 'params': {'temperature': 0.5}, 'metrics': {'accuracy': 0.8}}
        ]
        
        # Create temp file for save/load
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            
            try:
                # Save state
                self.tuner.save(temp_path)
                
                # Create new tuner with different state
                new_tuner = RLTuner(
                    config=self.config,
                    target_model=self.target_model,
                    validation_dataset=self.validation_dataset,
                    generator=self.generator
                )
                new_tuner.best_reward = 0.0
                
                # Load state
                new_tuner.load(temp_path)
                
                # Verify state was loaded
                self.assertEqual(new_tuner.best_params, self.tuner.best_params)
                self.assertEqual(new_tuner.best_reward, self.tuner.best_reward)
                self.assertEqual(len(new_tuner.history), len(self.tuner.history))
                
            finally:
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)


if __name__ == '__main__':
    unittest.main() 