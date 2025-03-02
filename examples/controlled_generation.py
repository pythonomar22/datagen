#!/usr/bin/env python
"""
Example of controlled generation with domain, style, and tone constraints in DataGen
"""

import os
import sys
import logging
import json

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results


def main():
    """Demonstrate controlled generation with specific constraints"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # For testing purposes, temporarily skip the API key check
    # and simulate execution even without an API key
    simulate_without_api = True
    
    # Check for API key before proceeding
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not simulate_without_api:
        print("\n⚠️  API key not found! ⚠️")
        print("Please set your OpenAI API key using one of these methods:")
        print("1. Set the OPENAI_API_KEY environment variable:")
        print("   export OPENAI_API_KEY=your-api-key")
        print("2. Or uncomment and update this line in the script:")
        print("   os.environ[\"OPENAI_API_KEY\"] = \"your-api-key\"")
        return
    
    # Create a base configuration
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    config.sampling.temperature = 0.7
    
    # Create a generator
    generator = Generator(config)
    
    # Define domain-specific constraints for different domains
    domains = [
        {
            "domain": "legal", 
            "keywords": ["law", "regulation", "compliance", "legal", "jurisdiction", "statute"],
            "style": "formal",
            "tone": "authoritative",
            "complexity": "high"
        },
        {
            "domain": "medical", 
            "keywords": ["health", "patient", "treatment", "diagnosis", "medical", "clinical"],
            "style": "precise",
            "tone": "informative",
            "complexity": "high"
        },
        {
            "domain": "finance", 
            "keywords": ["investment", "market", "financial", "assets", "portfolio", "risk"],
            "style": "analytical",
            "tone": "neutral",
            "complexity": "medium"
        },
        {
            "domain": "education", 
            "keywords": ["learning", "education", "student", "curriculum", "teaching", "assessment"],
            "style": "explanatory",
            "tone": "supportive",
            "complexity": "adjustable"
        }
    ]
    
    # List of base instructions to apply domain constraints to
    base_instructions = [
        "Write a short guide about ethical considerations.",
        "Explain the impact of technology on society.",
        "Describe best practices for documentation.",
        "Explain how to approach problem-solving."
    ]
    
    print("Generating domain-specific examples with controlled constraints...")
    
    # IMPORTANT NOTE: This is a demonstration of a feature that requires implementation
    print("\n⚠️ NOTE: The generate_with_constraints method is not yet implemented. ⚠️")
    print("This example demonstrates how controlled generation with constraints would work.")
    print("Below is a simulated output of what this feature would produce.")
    
    # Create simulated results for demonstration
    all_results = Results([])
    for domain_config in domains:
        domain_name = domain_config["domain"]
        print(f"\nSimulating examples for the {domain_name} domain...")
        
        # For each base instruction, create a simulated response for this domain
        for instruction in base_instructions:
            # Instead of actual generation, we'll create example data
            example = {
                "instruction": instruction,
                "constraints": {
                    "domain": domain_config["domain"],
                    "keywords": domain_config["keywords"],
                    "style": domain_config["style"],
                    "tone": domain_config["tone"],
                    "complexity": domain_config["complexity"]
                },
                "response": f"[This would be a {domain_config['style']} response about {instruction.lower()} in the {domain_name} domain, using keywords like {', '.join(domain_config['keywords'][:3])}, with a {domain_config['tone']} tone and {domain_config['complexity']} complexity level.]"
            }
            
            # Add to our collection
            all_results.data.append(example)
        
        print(f"Simulated {len(base_instructions)} examples for {domain_name} domain")
    
    # Print a summary
    print(f"\nTotal examples simulated: {len(all_results.data)}")
    
    # Show samples of simulated content for each domain
    print("\nSamples of simulated content:")
    
    # Group results by domain for display
    by_domain = {}
    for example in all_results.data:
        domain = example.get("constraints", {}).get("domain", "unknown")
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(example)
    
    # Display one example per domain
    for domain, examples in by_domain.items():
        if examples:
            print(f"\n--- {domain.upper()} DOMAIN EXAMPLE ---")
            example = examples[0]
            print(f"Instruction: {example.get('instruction', '')}")
            print(f"Constraints: {example.get('constraints', {})}")
            print(f"Response: {example.get('response', '')}")
    
    # Save the results to the current directory
    output_file = "domain_specific_data.jsonl"
    all_results.save(output_file)
    print(f"\nSaved domain-specific examples to {output_file}")
    
    # Implementation guidance
    print("\nTo implement this feature in the DataGen library:")
    print("1. Add a generate_with_constraints method to the Generator class")
    print("2. Create domain-specific prompt templates in the generation module")
    print("3. Implement constraint handling logic to enforce domain, style, tone, etc.")
    
    # Change back to false after testing
    simulate_without_api = False
    
    
if __name__ == "__main__":
    main() 