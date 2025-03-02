#!/usr/bin/env python
"""
Example of controlled generation with domain, style, and tone constraints in DataGen
"""

import os
import sys
import logging

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results


def main():
    """Demonstrate controlled generation with specific constraints"""
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
    all_results = Results([])  # Empty results to store all generated data
    
    # Generate examples for each domain and base instruction
    for domain_config in domains:
        domain_name = domain_config["domain"]
        print(f"\nGenerating examples for the {domain_name} domain...")
        
        # Set the controlled generation constraints
        constraints = {
            "domain": domain_config["domain"],
            "keywords": domain_config["keywords"],
            "style": domain_config["style"],
            "tone": domain_config["tone"],
            "complexity": domain_config["complexity"]
        }
        
        # Generate examples for this domain
        domain_results = generator.generate_with_constraints(
            instructions=base_instructions,
            constraints=constraints,
            examples_per_instruction=1
        )
        
        print(f"Generated {len(domain_results)} examples for {domain_name} domain")
        
        # Add to our collection
        all_results = all_results.extend(domain_results)
    
    # Print a summary
    print(f"\nTotal examples generated: {len(all_results)}")
    
    # Show samples of generated content for each domain
    print("\nSamples of generated content:")
    
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
            print(f"Response (excerpt): {example.get('response', '')[:200]}...")
    
    # Save the results
    output_file = "domain_specific_data.jsonl"
    all_results.save(output_file)
    print(f"\nSaved domain-specific controlled generation examples to {output_file}")
    
    
if __name__ == "__main__":
    main() 