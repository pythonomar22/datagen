#!/usr/bin/env python
"""
Test script to verify the examples would work with an API key
"""

import os
import sys
import logging
import tempfile
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datagen import Generator, Config, Results

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create a mock generator that doesn't make actual API calls
class MockGenerator(Generator):
    def __init__(self, config):
        print("Initialized MockGenerator with simulated API key")
        self.config = config
        self.quality_filter = MockQualityFilter()
    
    def generate_from_seed(self, seed_examples, count, method="self_instruct"):
        print(f"Simulating generation of {count} examples from {len(seed_examples)} seed examples")
        # Create mock results
        mock_data = []
        for i in range(count):
            mock_data.append({
                "instruction": f"This is simulated instruction {i}",
                "response": f"This is a simulated response for instruction {i}. The API call was mocked to test without using real API credits."
            })
        return Results(mock_data)

class MockQualityFilter:
    def __init__(self):
        self.filters = []
    
    def register_filter(self, filter_fn):
        self.filters.append(filter_fn)
        print(f"Registered filter: {filter_fn.__name__}")
    
    def filter(self, results):
        print(f"Applied {len(self.filters)} filter(s) to {len(results.data)} examples")
        return results

def test_basic_usage():
    # Set a mock API key
    os.environ["OPENAI_API_KEY"] = "fake-api-key-for-testing"
    
    # Create a config and mock generator
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    config.sampling.temperature = 0.7
    
    generator = MockGenerator(config)
    
    # Define seed examples
    seed_examples = [
        {
            "instruction": "Write a poem about AI.",
            "response": "In circuits deep where logic meets, A new form of thinking silently competes..."
        },
        {
            "instruction": "Explain how a refrigerator works.",
            "response": "A refrigerator works by removing heat from its interior compartment..."
        }
    ]
    
    # Generate synthetic data
    results = generator.generate_from_seed(
        seed_examples=seed_examples,
        count=5,
        method="self_instruct"
    )
    
    # Test custom filter registration
    def test_filter(example):
        return True
    
    generator.quality_filter.register_filter(test_filter)
    filtered_results = generator.quality_filter.filter(results)
    
    print(f"\nGenerated {len(results.data)} examples")
    print("First example:")
    if len(results.data) > 0:
        example = results.data[0]
        print(f"Instruction: {example.get('instruction', '')}")
        print(f"Response: {example.get('response', '')}")
    
    # Clean up
    del os.environ["OPENAI_API_KEY"]
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_basic_usage() 