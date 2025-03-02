#!/usr/bin/env python
"""
Basic example usage of the DataGen library
"""

import os
import sys
import logging

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results


def main():
    """Demonstrate basic usage of DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create a configuration
    config = Config()

    # Customize the configuration
    config.generation.model_name = "gpt-3.5-turbo"  # Change to your preferred model
    config.generation.backend = "openai"            # Change based on your API access
    config.sampling.temperature = 0.7
    config.quality.enable_filtering = True
    
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
    
    # Define seed examples (instruction-response pairs)
    seed_examples = [
        {
            "instruction": "Write a poem about artificial intelligence.",
            "response": "Silicon dreams and neural streams,\nIn the heart of the machine,\nA mind awakens, learns, and gleams,\nThinking thoughts once unforeseen.\n\nElectronic synapses firing bright,\nPatterns forming, day and night,\nLearning, growing, gaining sight,\nIn this new cognitive light.\n\nHuman-made but self-refined,\nArtificial yet defined,\nBy the bounds of its designed mind,\nAnd the data it's assigned.\n\nA partnership of human and code,\nTogether on this winding road,\nCarrying knowledge as our load,\nInto futures yet untold."
        },
        {
            "instruction": "Explain how a refrigerator works.",
            "response": "A refrigerator works by removing heat from its interior and transferring it to the external environment. This process uses a refrigeration cycle with these key components:\n\n1. Compressor: Compresses the refrigerant gas, raising its temperature and pressure.\n\n2. Condenser coils: Hot refrigerant flows through these coils (usually on the back or bottom), releasing heat to the surrounding air and cooling down.\n\n3. Expansion valve: The cooled, high-pressure liquid refrigerant passes through this valve, which reduces its pressure, causing it to expand and become much colder.\n\n4. Evaporator coils: The cold refrigerant flows through these coils inside the fridge, absorbing heat from the interior.\n\nAs the refrigerant absorbs heat, it evaporates back into a gas. This gas returns to the compressor, and the cycle repeats. The thermostat controls this cycle to maintain the desired temperature. This continuous heat transfer keeps the inside of the refrigerator cold."
        },
        {
            "instruction": "Give me a recipe for chocolate chip cookies.",
            "response": "# Classic Chocolate Chip Cookies\n\n## Ingredients:\n- 2¼ cups all-purpose flour\n- 1 teaspoon baking soda\n- 1 teaspoon salt\n- 1 cup (2 sticks) unsalted butter, softened\n- ¾ cup granulated sugar\n- ¾ cup packed brown sugar\n- 2 large eggs\n- 2 teaspoons vanilla extract\n- 2 cups semi-sweet chocolate chips\n- 1 cup chopped nuts (optional)\n\n## Instructions:\n\n1. Preheat oven to 375°F (190°C).\n\n2. In a small bowl, mix flour, baking soda, and salt.\n\n3. In a large mixing bowl, cream together the butter, granulated sugar, and brown sugar until smooth and fluffy.\n\n4. Beat in eggs one at a time, then stir in the vanilla.\n\n5. Gradually blend in the dry ingredients until just combined.\n\n6. Fold in the chocolate chips and nuts (if using).\n\n7. Drop by rounded tablespoons onto ungreased baking sheets, spacing cookies about 2 inches apart.\n\n8. Bake for 9-11 minutes or until golden brown around the edges.\n\n9. Cool on baking sheets for 2 minutes, then transfer to wire racks to cool completely.\n\nEnjoy with a glass of milk!"
        }
    ]
    
    # Generate synthetic data
    results = generator.generate_from_seed(
        seed_examples=seed_examples,
        count=10,
        method="self_instruct"
    )
    
    # Print generation information
    print(f"\nGenerated {len(results)} examples")
    print("\nGeneration Summary:")
    summary = results.summary()
    if 'instruction_length' in summary:
        il = summary['instruction_length']
        print(f"  Instruction length (min/avg/max): {il['min']}/{il['mean']:.1f}/{il['max']}")
        
    if 'response_length' in summary:
        rl = summary['response_length']
        print(f"  Response length (min/avg/max): {rl['min']}/{rl['mean']:.1f}/{rl['max']}")
    
    # Print a sample of the generated data
    print("\nSample of generated data:")
    for i, example in enumerate(results.data[:3]):  # Print first 3 examples
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Response: {example['response'][:100]}...")  # Print just the beginning
        
    # Save the results
    output_file = "generated_data.jsonl"
    results.save(output_file)
    print(f"\nSaved all generated data to {output_file}")
    
    
if __name__ == "__main__":
    main() 