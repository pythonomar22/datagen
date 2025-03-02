#!/usr/bin/env python
"""
Example of evolving instructions using the Evol-Instruct method
"""

import os
import sys
import logging

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results


def main():
    """Demonstrate instruction evolution with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration and enable evol-instruct
    config = Config()
    config.generation.evol_instruct = True
    config.generation.model_name = "gpt-3.5-turbo"  # Change to your preferred model
    config.generation.backend = "openai"            # Change based on your API access
    config.sampling.temperature = 0.7
    
    # Important: Set your API key
    # Either through environment variable: os.environ["OPENAI_API_KEY"] = "your-key" 
    # Or directly in the generator
    
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
    
    # Define some basic instructions to evolve
    instructions = [
        "Write a poem about nature.",
        "Explain how solar panels work.",
        "Give career advice for a college graduate.",
        "Write a recipe for pasta sauce.",
        "Explain the concept of inflation."
    ]
    
    # Set number of evolution rounds
    evolution_rounds = 2
    
    # Evolve the instructions
    print(f"Evolving {len(instructions)} instructions through {evolution_rounds} rounds...")
    results = generator.evolve_instructions(
        instructions=instructions,
        rounds=evolution_rounds
    )
    
    # Print evolution summary
    print(f"\nEvolved {len(results)} instructions")
    
    # Print examples of the evolution
    print("\nEvolution Examples:")
    
    # Group by original instruction
    evolution_chains = {}
    for item in results.data:
        original = item.get("original_instruction", "")
        if original not in evolution_chains:
            evolution_chains[original] = []
        evolution_chains[original].append(item)
    
    # Print each evolution chain
    for i, (original, chain) in enumerate(list(evolution_chains.items())[:3]):  # Show first 3 chains
        print(f"\nEvolution Chain {i+1}:")
        print(f"Original: {original}")
        
        # Sort by round
        sorted_chain = sorted(chain, key=lambda x: x.get("evolution_round", 0))
        
        for j, step in enumerate(sorted_chain):
            print(f"  Round {step.get('evolution_round', j+1)}: {step.get('evolved_instruction', '')}")
    
    # Save the results
    output_file = "evolved_instructions.jsonl"
    results.save(output_file)
    print(f"\nSaved all evolved instructions to {output_file}")
    
    
if __name__ == "__main__":
    main() 