#!/usr/bin/env python
"""
Example of using custom quality filters with DataGen
"""

import os
import sys
import logging
import re

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.quality.filter import QualityFilter


def main():
    """Demonstrate quality filtering with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration with custom quality settings
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    
    # Customize quality filtering settings
    config.quality.enable_filtering = True
    config.quality.min_instruction_length = 10
    config.quality.min_response_length = 50
    config.quality.similarity_threshold = 0.7  # Lower threshold to filter more aggressively
    
    # Check for API key before proceeding
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  API key not found! ⚠️")
        print("Please set your OpenAI API key using one of these methods:")
        print("1. Set the OPENAI_API_KEY environment variable:")
        print("   export OPENAI_API_KEY=your-api-key")
        print("2. Or uncomment and update this line in the script:")
        print("   os.environ[\"OPENAI_API_KEY\"] = \"your-api-key\"")
        return
    
    # Create a generator
    generator = Generator(config)
    
    # Define seed examples (deliberately including some low-quality ones)
    seed_examples = [
        {
            "instruction": "Explain quantum computing to a beginner.",
            "response": "Quantum computing uses quantum bits or qubits which can exist in multiple states at once, unlike classical bits that are either 0 or 1. This allows quantum computers to perform certain calculations much faster than traditional computers by exploiting phenomena like superposition and entanglement. While still in early stages, quantum computers show promise for solving complex problems in cryptography, material science, and optimization that are currently intractable."
        },
        {
            "instruction": "Short",  # Too short instruction
            "response": "This is a very short response."  # Too short response
        },
        {
            "instruction": "Write about machine learning.",
            "response": "Machine learning is a field of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions. Common approaches include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error)."
        },
        {
            "instruction": "Explain quantum computing simply.",  # Similar to first example
            "response": "Quantum computing is a type of computing that uses quantum bits or qubits. Unlike regular computers that use bits (0 or 1), quantum computers use qubits that can be both 0 and 1 at the same time. This makes them potentially much faster for certain problems."
        },
    ]
    
    # Define a custom filter function to reject examples containing certain keywords
    def filter_prohibited_keywords(example):
        """Filter out examples containing prohibited keywords"""
        prohibited_words = ["cryptocurrency", "illegal", "hack", "password"]
        
        if "instruction" in example:
            for word in prohibited_words:
                if word in example["instruction"].lower():
                    return False
                    
        if "response" in example:
            for word in prohibited_words:
                if word in example["response"].lower():
                    return False
                    
        return True
    
    # Register the custom filter with the generator's quality filter
    generator.quality_filter.register_filter(filter_prohibited_keywords)
    
    # Generate data
    print("Generating examples with quality filtering...")
    results = generator.generate_from_seed(
        seed_examples=seed_examples,
        count=10,
        method="self_instruct"
    )
    
    # Examine results
    print(f"\nGenerated and filtered: {len(results)} examples")
    
    # Show applied filters
    print("\nApplied filters:")
    for filter_fn in generator.quality_filter.filters:
        print(f"- {filter_fn.__name__}")
    
    # Demonstrate applying a filter after generation
    print("\nApplying additional filter after generation...")
    
    # Create a new filter to only keep examples with questions in the instruction
    def filter_question_instructions(example):
        """Keep only examples where the instruction contains a question"""
        if "instruction" in example:
            # Look for question marks or question words
            has_question_mark = "?" in example["instruction"]
            question_words = ["how", "what", "why", "when", "where", "who", "which"]
            has_question_word = any(word in example["instruction"].lower() for word in question_words)
            return has_question_mark or has_question_word
        return True
    
    # Apply the filter to results
    filtered_results = results.filter(filter_question_instructions)
    
    # Print filtered results
    print(f"After question filter: {len(filtered_results)}/{len(results)} examples kept")
    
    # Print sample of filtered results
    print("\nSample of filtered results:")
    for i, example in enumerate(filtered_results.data[:2]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response'][:100]}...")
    
    # Save filtered results
    output_file = "quality_filtered_data.jsonl"
    filtered_results.save(output_file)
    print(f"\nSaved filtered data to {output_file}")
    
    
if __name__ == "__main__":
    main() 