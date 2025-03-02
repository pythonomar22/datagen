#!/usr/bin/env python
"""
Example of using the DataGen command-line interface
"""

import os
import sys
import subprocess
import tempfile
import json
import logging
from pathlib import Path

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    """Run a shell command and print the output"""
    print(f"\n--- {description} ---")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command output:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Command failed with error:")
        print(e.stderr)
        return None


def main():
    """Demonstrate CLI usage with DataGen"""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a temporary directory for our example files
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Check if the datagen package is installed and accessible
        print("Checking if datagen CLI is available...")
        try:
            help_output = subprocess.run(
                ["python", "-m", "datagen.cli", "--help"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print("DataGen CLI is available")
        except subprocess.CalledProcessError:
            print("⚠️ DataGen CLI might not be installed correctly.")
            print("This example assumes you have installed the package or are running from the source directory.")
            return
        
        # Step 1: Create a configuration file
        config_path = os.path.join(temp_dir, "config.yaml")
        
        config = {
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1024
            },
            "quality": {
                "enable_filtering": True,
                "min_instruction_length": 5,
                "min_response_length": 20
            },
            "privacy": {
                "enable_privacy": False
            },
            "generation": {
                "model_name": "gpt-3.5-turbo",
                "backend": "openai"
            },
            "output_format": "jsonl",
            "log_level": "INFO"
        }
        
        # Write config to file
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        print(f"Created configuration file at {config_path}")
        
        # Step 2: Create seed examples
        seed_path = os.path.join(temp_dir, "seed_examples.jsonl")
        
        seed_examples = [
            {
                "instruction": "Explain quantum computing in simple terms.",
                "response": "Quantum computing uses the unique properties of quantum physics to process information in new ways. While regular computers use bits (0s and 1s), quantum computers use qubits that can exist in multiple states at once. This allows them to solve certain complex problems much faster than traditional computers."
            },
            {
                "instruction": "Write a short poem about technology.",
                "response": "Silicon dreams and digital streams,\nConnecting hearts across the miles,\nIn zeros, ones, and laser beams,\nOur modern world of tech beguiles.\n\nDevices smart and always near,\nCompanions in our daily quest,\nYet simple moments we hold dear,\nRemind us humans still know best."
            }
        ]
        
        # Write seed examples to file
        with open(seed_path, "w") as f:
            for example in seed_examples:
                f.write(json.dumps(example) + "\n")
                
        print(f"Created seed examples file at {seed_path}")
        
        # Step 3: Run the 'init' command to check setup
        init_cmd = [
            "python", "-m", "datagen.cli", "init",
            "--config", config_path,
            "--preset", "instruction_tuning",
            "--log-level", "INFO"
        ]
        
        run_command(init_cmd, "Initializing DataGen")
        
        # Step 4: Run the 'generate' command to create synthetic data
        output_path = os.path.join(temp_dir, "generated_data.jsonl")
        
        generate_cmd = [
            "python", "-m", "datagen.cli", "generate",
            "--config", config_path,
            "--seed-file", seed_path,
            "--output", output_path,
            "--count", "5",
            "--method", "self_instruct"
        ]
        
        # API key is required for generation - we'll skip actual execution
        # and just show the command
        print("\n--- Generating synthetic data (command demonstration only) ---")
        print(f"Command to run: {' '.join(generate_cmd)}")
        print("\nNote: This command requires an OpenAI API key to be set")
        print("You would set it with --api-key or OPENAI_API_KEY environment variable")
        print("Not executing the command in this example")
        
        # Step 5: Demonstrate the 'evolve' command
        instructions_path = os.path.join(temp_dir, "instructions.txt")
        evolved_path = os.path.join(temp_dir, "evolved_instructions.jsonl")
        
        # Create instructions file
        with open(instructions_path, "w") as f:
            f.write("Explain machine learning.\n")
            f.write("Write a story about robots.\n")
            f.write("Give tips for public speaking.\n")
        
        evolve_cmd = [
            "python", "-m", "datagen.cli", "evolve",
            "--config", config_path,
            "--instructions-file", instructions_path,
            "--output", evolved_path,
            "--rounds", "2"
        ]
        
        print("\n--- Evolving instructions (command demonstration only) ---")
        print(f"Command to run: {' '.join(evolve_cmd)}")
        print("\nNote: This command requires an OpenAI API key to be set")
        print("Not executing the command in this example")
        
        # Step 6: Demonstrate the 'prepare' command for ML training
        # Create a sample generated file for demonstration
        with open(output_path, "w") as f:
            # Write some dummy data
            for i in range(10):
                example = {
                    "instruction": f"Sample instruction {i}",
                    "response": f"Sample response {i} with enough content to make it valid."
                }
                f.write(json.dumps(example) + "\n")
                
        prepare_cmd = [
            "python", "-m", "datagen.cli", "prepare",
            "--input-file", output_path,
            "--output-dir", os.path.join(temp_dir, "training_data"),
            "--split",
            "--train-ratio", "0.8",
            "--format", "jsonl"
        ]
        
        run_command(prepare_cmd, "Preparing data for model training")
        
        # List all created files
        print("\n--- Files created during this example ---")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        # Uncomment to delete the temp directory when you're done testing
        # import shutil
        # shutil.rmtree(temp_dir)
        print("Note: Temporary directory was NOT deleted so you can examine the files")
        print(f"Please manually delete {temp_dir} when you're done")
    
    
if __name__ == "__main__":
    main() 