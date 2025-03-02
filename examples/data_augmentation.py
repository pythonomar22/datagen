#!/usr/bin/env python
"""
Example of augmenting existing datasets with DataGen
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.pipeline.io import DataLoader, DataExporter


def main():
    """Demonstrate data augmentation with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

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
    
    # Load an existing dataset
    # For this example, we'll create a small "existing" dataset
    print("Creating a sample 'existing' dataset...")
    
    original_data = [
        {
            "instruction": "Explain the concept of recursion in programming.",
            "response": "Recursion in programming is a technique where a function calls itself to solve a problem. It's particularly useful for tasks that can be broken down into similar subtasks. A recursive function typically has a base case that stops the recursion and a recursive case that continues it. For example, calculating a factorial: 5! can be calculated as 5 × 4!, which is 5 × 4 × 3!, and so on until we reach the base case of 1! = 1."
        },
        {
            "instruction": "What is photosynthesis and why is it important?",
            "response": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose (sugar). They use carbon dioxide and water, releasing oxygen as a byproduct. This process is crucial because it provides the oxygen we breathe, forms the base of food chains by creating energy-rich compounds, and helps regulate atmospheric carbon dioxide levels, which affects climate."
        },
        {
            "instruction": "How does cloud computing work?",
            "response": "Cloud computing delivers various services over the internet, including storage, databases, software, and processing power. Instead of keeping files and applications on local devices, they're stored remotely on servers owned by providers like AWS, Google Cloud, or Microsoft Azure. Users access these resources on-demand over the internet, paying only for what they use. This model provides scalability, flexibility, and cost efficiency compared to maintaining physical infrastructure."
        }
    ]
    
    # Save the original dataset
    original_file = "original_dataset.jsonl"
    with open(original_file, "w") as f:
        for example in original_data:
            f.write(json.dumps(example) + "\n")
            
    print(f"Saved 'existing' dataset with {len(original_data)} examples to {original_file}")
    
    # Create a configuration for the generator
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    config.sampling.temperature = 0.8  # Slightly higher temperature for more diverse augmentation
    
    # Create a generator
    generator = Generator(config)
    
    print("\nAugmenting the dataset using different techniques...")
    
    # 1. Paraphrasing: Generate variations of existing examples
    print("\n--- Technique 1: Paraphrasing ---")
    paraphrased_results = generator.augment_by_paraphrasing(
        examples=original_data,
        variations_per_example=2,
        fields_to_paraphrase=["instruction"]  # Only paraphrase instructions
    )
    
    print(f"Generated {len(paraphrased_results)} paraphrased examples")
    
    # Print example of paraphrasing
    if len(paraphrased_results) > 0:
        original = original_data[0]["instruction"]
        variations = [
            example["instruction"] 
            for example in paraphrased_results.data 
            if example.get("original_id") == 0
        ]
        
        print("\nOriginal instruction:")
        print(f"  {original}")
        print("\nParaphrased variations:")
        for i, variation in enumerate(variations):
            print(f"  {i+1}. {variation}")
    
    # 2. Style variation: Generate examples in different styles/formats
    print("\n--- Technique 2: Style Variation ---")
    style_variations = generator.augment_with_style_variation(
        examples=original_data,
        styles=["formal", "conversational", "technical"],
        examples_per_style=1
    )
    
    print(f"Generated {len(style_variations)} style variations")
    
    # Print example of style variations
    if len(style_variations) > 0:
        print("\nStyle variation examples:")
        
        # Group by style
        by_style = {}
        for example in style_variations.data:
            style = example.get("style", "unknown")
            if style not in by_style:
                by_style[style] = []
            by_style[style].append(example)
        
        # Show one example per style
        for style, examples in by_style.items():
            if examples:
                print(f"\nStyle: {style.upper()}")
                print(f"Instruction: {examples[0].get('instruction', '')}")
                print(f"Response: {examples[0].get('response', '')[:150]}...")
    
    # 3. Domain transfer: Adapt examples to different domains
    print("\n--- Technique 3: Domain Transfer ---")
    domain_transfer = generator.augment_with_domain_transfer(
        examples=original_data,
        target_domains=["healthcare", "finance", "education"],
        examples_per_domain=1
    )
    
    print(f"Generated {len(domain_transfer)} domain-transferred examples")
    
    # Print example of domain transfer
    if len(domain_transfer) > 0:
        print("\nDomain transfer examples:")
        
        # Group by domain
        by_domain = {}
        for example in domain_transfer.data:
            domain = example.get("domain", "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(example)
        
        # Show one example per domain
        for domain, examples in by_domain.items():
            if examples:
                print(f"\nDomain: {domain.upper()}")
                print(f"Original concept: {examples[0].get('original_concept', '')}")
                print(f"Instruction: {examples[0].get('instruction', '')}")
                print(f"Response: {examples[0].get('response', '')[:150]}...")
    
    # 4. Generate similar but new examples
    print("\n--- Technique 4: Similar New Examples ---")
    similar_examples = generator.generate_from_seed(
        seed_examples=original_data,
        count=5,
        method="self_instruct"
    )
    
    print(f"Generated {len(similar_examples)} new examples similar to the original dataset")
    
    # Print examples of similar new examples
    if len(similar_examples) > 0:
        print("\nNew similar examples:")
        for i, example in enumerate(similar_examples.data[:2]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {example.get('instruction', '')}")
            print(f"Response: {example.get('response', '')[:150]}...")
    
    # Combine all augmented datasets
    print("\n--- Combining All Augmented Data ---")
    
    # First convert to Results object if any are plain lists
    original_results = Results(original_data)
    
    # Combine all results
    combined_results = original_results.extend(paraphrased_results)
    combined_results = combined_results.extend(style_variations)
    combined_results = combined_results.extend(domain_transfer)
    combined_results = combined_results.extend(similar_examples)
    
    print(f"Original dataset: {len(original_data)} examples")
    print(f"After augmentation: {len(combined_results)} examples")
    print(f"Augmentation ratio: {len(combined_results) / len(original_data):.1f}x")
    
    # Save the augmented dataset
    augmented_file = "augmented_dataset.jsonl"
    combined_results.save(augmented_file)
    print(f"\nSaved augmented dataset to {augmented_file}")
    
    # Analyze diversity of augmented dataset
    print("\n--- Diversity Analysis ---")
    
    # Here we would typically use metrics from the quality module
    # For this example, we'll just do a simple analysis
    
    # Count unique instructions
    instructions = [example.get("instruction", "") for example in combined_results.data]
    unique_instructions = len(set(instructions))
    
    print(f"Unique instructions: {unique_instructions}/{len(combined_results)} "
          f"({100 * unique_instructions / len(combined_results):.1f}%)")
    
    # Average instruction length
    avg_instruction_length = sum(len(instr) for instr in instructions) / len(instructions)
    print(f"Average instruction length: {avg_instruction_length:.1f} characters")
    
    print("\nData augmentation complete!")
    
    
if __name__ == "__main__":
    main() 