#!/usr/bin/env python
"""
Example of using the data pipeline features in DataGen
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.pipeline.io import DataLoader, DataExporter


def main():
    """Demonstrate data pipeline integration with DataGen"""
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
    
    # Set up output directories in the current directory
    output_dir = "pipeline_output"
    data_dir = os.path.join(output_dir, "data")
    model_dir = os.path.join(output_dir, "model_data")
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Created output directories:")
    print(f"- {output_dir}/")
    print(f"- {data_dir}/")
    print(f"- {model_dir}/")
    
    try:
        # Step 1: Create and save some seed data
        print("\n--- Step 1: Creating seed data ---")
        seed_data = [
            {
                "instruction": "Explain the concept of neural networks.",
                "response": "Neural networks are computing systems inspired by the biological neural networks in human brains. They consist of layers of interconnected nodes or 'neurons' that process information. Each connection can transmit a signal from one neuron to another, activated by the input it receives. The network learns by adjusting the strengths of these connections based on the data it processes."
            },
            {
                "instruction": "What is the difference between machine learning and deep learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve without explicit programming. Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep'). While traditional machine learning requires feature extraction by humans, deep learning can automatically discover the representations needed for detection or classification from raw data."
            }
        ]
        
        # Save in different formats for demonstration
        seed_jsonl_path = os.path.join(data_dir, "seed_data.jsonl")
        seed_json_path = os.path.join(data_dir, "seed_data.json")
        seed_csv_path = os.path.join(data_dir, "seed_data.csv")
        
        # Create Results from seed data
        seed_results = Results(seed_data)
        
        # Save in different formats
        seed_results.save(seed_jsonl_path, format="jsonl")
        seed_results.save(seed_json_path, format="json")
        seed_results.save(seed_csv_path, format="csv")
        
        print(f"Saved seed data in multiple formats:")
        print(f"JSONL: {seed_jsonl_path}")
        print(f"JSON: {seed_json_path}")
        print(f"CSV: {seed_csv_path}")
        
        # Step 2: Load data using DataLoader
        print("\n--- Step 2: Loading data with DataLoader ---")
        
        # Load from different formats
        jsonl_data = DataLoader.load_jsonl(seed_jsonl_path)
        json_data = DataLoader.load_json(seed_json_path)
        csv_data = DataLoader.load_csv(seed_csv_path)
        
        print(f"Loaded {len(jsonl_data)} examples from JSONL")
        print(f"Loaded {len(json_data)} examples from JSON")
        print(f"Loaded {len(csv_data)} examples from CSV")
        
        # Load all data from directory
        all_data = DataLoader.load_directory(data_dir)
        print(f"Loaded data from {len(all_data)} files in directory")
        
        # Step 3: Generate synthetic data
        print("\n--- Step 3: Generating synthetic data ---")
        
        config = Config()
        config.generation.model_name = "gpt-3.5-turbo"
        config.generation.backend = "openai"
        
        generator = Generator(config)
        
        results = generator.generate_from_seed(
            seed_examples=jsonl_data,  # Use the data we loaded
            count=5,
            method="self_instruct"
        )
        
        print(f"Generated {len(results)} examples")
        
        # Step 4: Export data for model training
        print("\n--- Step 4: Exporting data for model training ---")
        
        # Export with train/val split
        exported_files = DataExporter.export_for_model_training(
            results=results,
            output_dir=model_dir,
            split=True,
            train_ratio=0.8,
            format="jsonl"
        )
        
        print("Exported files for model training:")
        for file_type, file_path in exported_files.items():
            print(f"- {file_type}: {file_path}")
            
        # Step 5: Convert to DataFrame and demonstrate operations
        print("\n--- Step 5: Converting to DataFrame ---")
        
        df = results.to_dataframe()
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        if not df.empty:
            print("\nDataFrame columns:")
            for col in df.columns:
                print(f"- {col}")
                
            # Show some stats if instruction column exists
            if 'instruction' in df.columns:
                instruction_lengths = df['instruction'].str.len()
                print(f"\nInstruction length stats:")
                print(f"- Min: {instruction_lengths.min()}")
                print(f"- Max: {instruction_lengths.max()}")
                print(f"- Mean: {instruction_lengths.mean():.1f}")
                
        # Step 6: Manipulate and combine results
        print("\n--- Step 6: Manipulating Results objects ---")
        
        # Sample a subset of results
        sampled_results = results.sample(2, seed=42)
        print(f"Sampled {len(sampled_results)} examples from results")
        
        # Create a new result set
        additional_data = [
            {
                "instruction": "What is transfer learning?",
                "response": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It's popular in deep learning because it allows us to train deep neural networks with less data."
            }
        ]
        
        additional_results = Results(additional_data)
        
        # Combine results
        combined_results = results.extend(additional_results)
        print(f"Combined results now has {len(combined_results)} examples")
        
        # Save the final combined dataset
        final_output = os.path.join(output_dir, "final_dataset.jsonl")
        combined_results.save(final_output)
        print(f"\nSaved final combined dataset to {final_output}")
        
        # List all created files
        print("\n--- Files created during this example ---")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    print("\nData pipeline example complete!")
    print(f"All output files are in the '{output_dir}' directory")
    
    
if __name__ == "__main__":
    main() 