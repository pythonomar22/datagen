#!/usr/bin/env python
"""
Example of using templates for generating structured data with DataGen
"""

import os
import sys
import logging

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results


def main():
    """Demonstrate template-based generation with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"  # Change to your preferred model
    config.generation.backend = "openai"            # Change based on your API access
    config.sampling.temperature = 0.7
    
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
    
    # Define a template with variable placeholders
    template = """
    Write a short blog post about {topic} with a focus on {aspect}.
    The tone should be {tone} and the target audience is {audience}.
    The blog post should be approximately {length} words.

    Title:
    """
    
    # Define sets of variables to fill the template
    variables = [
        {
            "topic": "artificial intelligence",
            "aspect": "ethical considerations",
            "tone": "thoughtful and balanced",
            "audience": "technology professionals",
            "length": "400"
        },
        {
            "topic": "sustainable gardening",
            "aspect": "water conservation",
            "tone": "enthusiastic and practical",
            "audience": "home gardeners",
            "length": "300"
        },
        {
            "topic": "remote work",
            "aspect": "maintaining team culture",
            "tone": "professional but friendly",
            "audience": "managers and team leads",
            "length": "350"
        },
        {
            "topic": "fitness trends",
            "aspect": "accessibility and inclusivity",
            "tone": "motivational",
            "audience": "general public",
            "length": "250"
        }
    ]
    
    # Custom sampling parameters for template generation
    sampling_params = {
        "temperature": 0.8,  # Slightly more creative
        "max_tokens": 500    # Longer response
    }
    
    # Generate content from template
    print(f"Generating {len(variables)} blog posts from template...")
    results = generator.generate_from_template(
        template=template,
        variables=variables,
        custom_sampling_params=sampling_params
    )
    
    # Print generation summary
    print(f"\nGenerated {len(results)} blog posts")
    
    # Print examples
    print("\nGeneration Examples:")
    for i, item in enumerate(results.data):
        print(f"\nBlog Post {i + 1}:")
        print(f"Topic: {item['variables']['topic']}")
        print(f"Aspect: {item['variables']['aspect']}")
        print(f"Tone: {item['variables']['tone']}")
        print(f"Audience: {item['variables']['audience']}")
        print("\nGenerated Content:")
        print(f"{item['completion'][:200]}...")  # Print just the beginning
    
    # Save the results
    output_file = "generated_blog_posts.jsonl"
    results.save(output_file)
    print(f"\nSaved all generated content to {output_file}")
    
    
if __name__ == "__main__":
    main() 