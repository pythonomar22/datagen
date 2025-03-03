"""
Integration test for the RL-guided data generation optimization process.
"""

import os
import sys
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.rl_tuning import RLTuner


class SimpleClassificationOptimizationTest(unittest.TestCase):
    """
    Test optimization of data generation for a simple text classification task.
    
    This test creates a simple mock generator that produces synthetic data with 
    controllable class imbalance. The test verifies that the RLTuner can find
    parameters that improve the generator's output for a classification task.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple classification validation dataset
        self.validation_data = [
            {"text": "This is positive content that is good and nice", "label": 1},
            {"text": "Also positive happy content that is great", "label": 1},
            {"text": "More positive wonderful content", "label": 1},
            {"text": "This is negative bad content that is terrible", "label": 0},
            {"text": "Another bad negative example of content", "label": 0},
            {"text": "Very negative and terrible content", "label": 0}
        ]
        
        # Create a mock generator that produces data with controllable class balance
        # based on temperature
        self.mock_generator = MockGenerator()
        
        # Create config
        self.config = Config()
        self.config.rl_tuning.enable_rl_tuning = True
        self.config.rl_tuning.num_iterations = 3
        self.config.rl_tuning.batch_size = 10
        self.config.rl_tuning.rl_algorithm = "random_search"
        self.config.rl_tuning.reward_metric = "accuracy"
        
        # Initialize the tuner
        self.tuner = RLTuner(
            config=self.config,
            target_model=self.evaluate_classification_model,
            validation_dataset=self.validation_data,
            generator=self.mock_generator
        )
    
    def evaluate_classification_model(self, synthetic_examples):
        """
        Train a simple text classification model on synthetic data and evaluate it.
        
        Args:
            synthetic_examples: List of dictionaries with 'text' and 'label' fields
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if we have enough examples
        if len(synthetic_examples) < 2:
            return {"accuracy": 0, "f1": 0}
        
        # Extract text and labels
        texts = []
        labels = []
        
        for example in synthetic_examples:
            if "text" in example and "label" in example:
                texts.append(example["text"])
                labels.append(example["label"])
        
        # Check if we have enough examples after filtering
        if len(texts) < 2 or len(set(labels)) < 2:
            return {"accuracy": 0, "f1": 0}
        
        # Create feature vectors
        vectorizer = TfidfVectorizer(max_features=100)
        X_train = vectorizer.fit_transform(texts)
        y_train = np.array(labels)
        
        # Extract validation features
        val_texts = [item["text"] for item in self.validation_data]
        val_labels = [item["label"] for item in self.validation_data]
        X_val = vectorizer.transform(val_texts)
        y_val = np.array(val_labels)
        
        # Train a simple model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions)
        
        # Return metrics
        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "examples_used": len(texts),
            "class_distribution": f"{sum(y_train == 1)}/{sum(y_train == 0)}"
        }
    
    def test_optimization_improves_performance(self):
        """Test that RL optimization improves model performance."""
        # Set a baseline with unbalanced synthetic data
        mock_unbalanced_data = self.mock_generator.generate_imbalanced(
            class_ratio=0.9,  # 90% positive, 10% negative
            count=20
        )
        baseline_metrics = self.evaluate_classification_model(mock_unbalanced_data)
        baseline_accuracy = baseline_metrics["accuracy"]
        
        # Run RL optimization
        results = self.tuner.train()
        
        # Generate data with optimized parameters
        optimized_data = self.tuner.generate_with_best_params(count=20)
        optimized_metrics = self.evaluate_classification_model(optimized_data.data)
        optimized_accuracy = optimized_metrics["accuracy"]
        
        # Verify optimization improved performance
        print(f"\nBaseline accuracy: {baseline_accuracy:.4f}")
        print(f"Optimized accuracy: {optimized_accuracy:.4f}")
        print(f"Improvement: {optimized_accuracy - baseline_accuracy:.4f}")
        
        # Assert optimization found better parameters
        self.assertGreaterEqual(optimized_accuracy, baseline_accuracy)
        

class MockGenerator:
    """Mock generator that produces synthetic data with controllable class balance."""
    
    def __init__(self):
        """Initialize the mock generator."""
        self.config = Config()
        
    def generate_from_seed(self, seed_examples=None, count=10, method="self_instruct"):
        """
        Generate synthetic examples with class balance determined by temperature.
        
        At high temperature (1.0), the generator produces all positive examples.
        At low temperature (0.0), the generator produces all negative examples.
        In between, it produces a mixture based on the temperature.
        """
        # Use temperature to control class balance
        # Temperature 1.0 -> all positive examples
        # Temperature 0.0 -> all negative examples
        temperature = self.config.sampling.temperature
        positive_ratio = temperature  # 0.0 to 1.0
        
        # Generate examples with the specified class distribution
        examples = self.generate_imbalanced(positive_ratio, count)
        
        return Results(examples)
    
    def evolve_instructions(self, instructions=None, rounds=1):
        """Mock evolution that does the same as generate_from_seed."""
        return self.generate_from_seed(None, len(instructions) * 2)
    
    def generate_imbalanced(self, class_ratio, count):
        """Generate examples with specified class balance."""
        examples = []
        
        # Calculate how many positive examples to generate
        num_positive = int(count * class_ratio)
        num_negative = count - num_positive
        
        # Generate positive examples
        for i in range(num_positive):
            examples.append({
                "text": f"This is positive sample {i} with good content that is nice",
                "label": 1
            })
        
        # Generate negative examples
        for i in range(num_negative):
            examples.append({
                "text": f"This is negative sample {i} with bad content that is terrible",
                "label": 0
            })
        
        return examples


if __name__ == "__main__":
    unittest.main() 