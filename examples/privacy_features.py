#!/usr/bin/env python
"""
Example of using privacy preservation features in DataGen
"""

import os
import sys
import logging

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.privacy.privacy_manager import PrivacyManager


def main():
    """Demonstrate privacy preservation with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration with privacy settings enabled
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    
    # Enable privacy features
    config.privacy.enable_privacy = True
    config.privacy.differential_privacy = True
    config.privacy.dp_epsilon = 1.0  # Lower epsilon = more privacy (but more noise)
    config.privacy.enable_content_filtering = True
    
    # Add custom sensitive terms
    config.privacy.sensitive_terms = [
        "social security",
        "credit card",
        "passport",
        "password",
        "secret",
        "private"
    ]
    
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
    
    # Create seed examples with some sensitive information
    seed_examples = [
        {
            "instruction": "Write a short story about a detective solving a mystery.",
            "response": "Detective Morgan looked at the case file on her desk. The victim's password was '12345' and their social security number was 123-45-6789. She needed to find who had access to this sensitive information."
        },
        {
            "instruction": "Explain how to keep personal information secure online.",
            "response": "To keep your information secure, never share your passwords or credit card numbers. Use a password manager to generate strong, unique passwords for each site. Enable two-factor authentication when available. Regularly check your credit report for suspicious activity."
        },
        {
            "instruction": "What are the benefits of a passport?",
            "response": "A passport allows international travel and serves as official government identification. It proves citizenship and gives you consular protection when abroad. Many countries require passports to be valid for at least six months beyond your trip."
        }
    ]
    
    # Generate data with privacy protection
    print("Generating examples with privacy protection...")
    results = generator.generate_from_seed(
        seed_examples=seed_examples,
        count=5,
        method="self_instruct"
    )
    
    # Save the results
    privacy_output = "privacy_protected_data.jsonl"
    results.save(privacy_output)
    print(f"\nSaved privacy-protected data to {privacy_output}")
    
    # To demonstrate, let's also generate examples without privacy protection
    print("\nNow generating examples WITHOUT privacy protection for comparison...")
    
    # Create a new config without privacy
    config_no_privacy = Config()
    config_no_privacy.generation.model_name = config.generation.model_name
    config_no_privacy.generation.backend = config.generation.backend
    config_no_privacy.privacy.enable_privacy = False
    
    # Create a new generator without privacy
    generator_no_privacy = Generator(config_no_privacy)
    
    # Generate without privacy protection
    results_no_privacy = generator_no_privacy.generate_from_seed(
        seed_examples=seed_examples,
        count=5,
        method="self_instruct"
    )
    
    # Save the non-private results
    no_privacy_output = "non_private_data.jsonl"
    results_no_privacy.save(no_privacy_output)
    print(f"Saved non-private data to {no_privacy_output}")
    
    # Compare and demonstrate privacy
    print("\nPrivacy comparison:")
    print("===================")
    
    # Apply privacy to the non-private results to see the difference
    print("\nApplying privacy processing to the non-private data...")
    privacy_manager = PrivacyManager(config.privacy)
    protected_results = privacy_manager.process(results_no_privacy)
    
    # Print a sample to show the difference
    if len(protected_results) > 0:
        print("\nSample before and after privacy protection:")
        idx = min(0, len(protected_results) - 1)  # Get first example or last if empty
        
        print("\nBEFORE PRIVACY PROTECTION:")
        print(f"Instruction: {results_no_privacy.data[idx].get('instruction', '')}")
        print(f"Response (excerpt): {results_no_privacy.data[idx].get('response', '')[:200]}...")
        
        print("\nAFTER PRIVACY PROTECTION:")
        print(f"Instruction: {protected_results.data[idx].get('instruction', '')}")
        print(f"Response (excerpt): {protected_results.data[idx].get('response', '')[:200]}...")
    
    # Save the post-processed results
    post_privacy_output = "post_processed_privacy_data.jsonl"
    protected_results.save(post_privacy_output)
    print(f"\nSaved post-processed privacy data to {post_privacy_output}")
    
    
if __name__ == "__main__":
    main() 