"""
CLI commands for DataGen
"""

import os
import sys
import logging
import json
from typing import List, Dict, Any, Optional
import click
from pathlib import Path

from datagen.config import Config
from datagen.generator import Generator
from datagen.results import Results
from datagen.pipeline.io import DataLoader, DataExporter
from datagen import __version__

logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Set up logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
        
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
@click.version_option(__version__)
def cli():
    """DataGen: Synthetic Data Generation for LLM Training"""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--preset', '-p', type=str, 
              help='Configuration preset to use')
@click.option('--backend', type=str, default=None,
              help='Generation backend (e.g., openai, anthropic, huggingface)')
@click.option('--model', type=str, default=None,
              help='Model to use for generation')
@click.option('--api-key', type=str, default=None,
              help='API key for the backend service')
@click.option('--log-level', type=str, default='INFO',
              help='Logging level')
def init(config, preset, backend, model, api_key, log_level):
    """Initialize a new configuration file"""
    setup_logging(log_level)
    
    if config and os.path.exists(config):
        click.echo(f"Loading existing configuration from {config}")
        user_config = Config.from_yaml(config)
    else:
        user_config = Config()
        
    # Apply preset if specified
    if preset:
        try:
            user_config = user_config.get_preset(preset)
            click.echo(f"Applied preset: {preset}")
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
            
    # Override with command-line options
    if backend:
        user_config.generation.backend = backend
        
    if model:
        user_config.generation.model_name = model
        
    if api_key:
        provider = backend or user_config.generation.backend
        user_config.api_keys[provider] = api_key
        
    # Save the configuration
    if not config:
        config = 'datagen_config.yaml'
        
    user_config.save(config)
    click.echo(f"Configuration saved to {config}")
    

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Path to configuration file')
@click.option('--seed-file', '-s', type=click.Path(exists=True), required=True,
              help='Path to seed examples file')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Path to output file')
@click.option('--count', '-n', type=int, default=100,
              help='Number of examples to generate')
@click.option('--method', type=click.Choice(['self_instruct', 'evol_instruct']), 
              default='self_instruct', 
              help='Generation method to use')
@click.option('--api-key', type=str, default=None,
              help='API key for the backend service')
@click.option('--format', type=click.Choice(['jsonl', 'json', 'csv']), default=None,
              help='Output format')
@click.option('--log-level', type=str, default='INFO',
              help='Logging level')
def generate(config, seed_file, output, count, method, api_key, format, log_level):
    """Generate synthetic data from seed examples"""
    setup_logging(log_level)
    
    # Load configuration
    user_config = Config.from_yaml(config)
    
    # Set API key if provided
    if api_key:
        provider = user_config.generation.backend
        user_config.api_keys[provider] = api_key
        
    # Create generator
    generator = Generator(user_config)
    
    # Check if the generator supports the requested method
    if method not in generator.generators:
        click.echo(f"Error: Method '{method}' is not enabled in the configuration.", err=True)
        sys.exit(1)
        
    # Load seed examples
    try:
        seed_examples = DataLoader.load(seed_file)
        click.echo(f"Loaded {len(seed_examples)} seed examples from {seed_file}")
    except Exception as e:
        click.echo(f"Error loading seed examples: {e}", err=True)
        sys.exit(1)
        
    # Generate data
    click.echo(f"Generating {count} examples using {method}...")
    results = generator.generate_from_seed(
        seed_examples=seed_examples,
        count=count,
        method=method
    )
    
    # Save results
    try:
        results.save(output, format)
        click.echo(f"Generated {len(results)} examples, saved to {output}")
        
        # Print a summary
        summary = results.summary()
        click.echo("\nGeneration Summary:")
        click.echo(f"  Method: {method}")
        click.echo(f"  Generated examples: {summary['count']}")
        
        if 'instruction_length' in summary:
            instr_len = summary['instruction_length']
            click.echo(f"  Instruction length (min/avg/max): {instr_len['min']}/{instr_len['mean']:.1f}/{instr_len['max']}")
            
        if 'response_length' in summary:
            resp_len = summary['response_length']
            click.echo(f"  Response length (min/avg/max): {resp_len['min']}/{resp_len['mean']:.1f}/{resp_len['max']}")
            
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Path to configuration file')
@click.option('--instructions-file', '-i', type=click.Path(exists=True), required=True,
              help='Path to file containing instructions to evolve')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Path to output file')
@click.option('--rounds', '-r', type=int, default=1,
              help='Number of evolution rounds')
@click.option('--api-key', type=str, default=None,
              help='API key for the backend service')
@click.option('--format', type=click.Choice(['jsonl', 'json', 'csv']), default=None,
              help='Output format')
@click.option('--log-level', type=str, default='INFO',
              help='Logging level')
def evolve(config, instructions_file, output, rounds, api_key, format, log_level):
    """Evolve instructions into more complex versions"""
    setup_logging(log_level)
    
    # Load configuration and enable evol-instruct
    user_config = Config.from_yaml(config)
    user_config.generation.evol_instruct = True
    
    # Set API key if provided
    if api_key:
        provider = user_config.generation.backend
        user_config.api_keys[provider] = api_key
        
    # Create generator
    generator = Generator(user_config)
    
    # Load instructions
    try:
        data = DataLoader.load(instructions_file)
        instructions = []
        
        # Extract instructions from the data
        for item in data:
            if 'instruction' in item:
                instructions.append(item['instruction'])
            elif 'text' in item:
                instructions.append(item['text'])
            elif isinstance(item, str):
                instructions.append(item)
                
        if not instructions:
            # Try reading as plain text file with one instruction per line
            with open(instructions_file, 'r') as f:
                instructions = [line.strip() for line in f if line.strip()]
                
        click.echo(f"Loaded {len(instructions)} instructions from {instructions_file}")
        
        if not instructions:
            click.echo("Error: No instructions found in input file", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error loading instructions: {e}", err=True)
        sys.exit(1)
        
    # Evolve instructions
    click.echo(f"Evolving {len(instructions)} instructions through {rounds} rounds...")
    results = generator.evolve_instructions(
        instructions=instructions,
        rounds=rounds
    )
    
    # Save results
    try:
        results.save(output, format)
        click.echo(f"Evolved {len(results)} instructions, saved to {output}")
    except Exception as e:
        click.echo(f"Error saving results: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True,
              help='Path to input file with generated examples')
@click.option('--output-dir', '-o', type=click.Path(), required=True,
              help='Directory to save the processed datasets')
@click.option('--split/--no-split', default=True,
              help='Whether to split into train/val sets')
@click.option('--train-ratio', type=float, default=0.8,
              help='Ratio of data to use for training (if splitting)')
@click.option('--format', '-f', type=click.Choice(['jsonl', 'json', 'csv']), default='jsonl',
              help='Output format')
@click.option('--log-level', type=str, default='INFO',
              help='Logging level')
def prepare(input_file, output_dir, split, train_ratio, format, log_level):
    """Prepare synthetic data for model training"""
    setup_logging(log_level)
    
    # Load the generated data
    try:
        results = Results.load(input_file)
        click.echo(f"Loaded {len(results)} examples from {input_file}")
    except Exception as e:
        click.echo(f"Error loading input file: {e}", err=True)
        sys.exit(1)
        
    # Prepare for model training
    try:
        output_files = DataExporter.export_for_model_training(
            results=results,
            output_dir=output_dir,
            format=format,
            split=split,
            train_ratio=train_ratio
        )
        
        click.echo(f"Prepared data for model training:")
        for name, path in output_files.items():
            click.echo(f"  {name}: {path}")
            
    except Exception as e:
        click.echo(f"Error preparing data: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI"""
    cli()
    
    
if __name__ == '__main__':
    main() 