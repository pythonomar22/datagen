#!/usr/bin/env python
"""
Example of evaluating the quality of generated datasets with DataGen
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.pipeline.io import DataLoader
from datagen.quality.metrics import DatasetMetrics


def main():
    """Demonstrate dataset evaluation with quality metrics"""
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
    
    # Create a sample dataset for evaluation
    print("Creating sample datasets for evaluation...")
    
    # Define seed examples for different domains
    general_examples = [
        {
            "instruction": "Explain how rainbows form.",
            "response": "Rainbows form when sunlight enters water droplets in the air, gets refracted, reflects off the back of the droplet, and refracts again as it exits. This process separates the light into its component colors, creating the characteristic arc of red, orange, yellow, green, blue, indigo, and violet that we see as a rainbow."
        },
        {
            "instruction": "What are the main causes of climate change?",
            "response": "The main causes of climate change are greenhouse gas emissions from burning fossil fuels (like coal, oil, and natural gas), deforestation (which reduces CO2 absorption), industrial processes, and agricultural practices. These activities increase the concentration of greenhouse gases in our atmosphere, trapping more heat and leading to global warming and various climate disruptions."
        }
    ]
    
    technical_examples = [
        {
            "instruction": "Explain how neural networks learn.",
            "response": "Neural networks learn through a process called backpropagation. First, the network makes predictions using current weights. Then it calculates the error between predictions and actual values using a loss function. This error is propagated backward through the network, and gradients are computed to determine how each weight contributes to the error. Finally, weights are updated using an optimization algorithm (like gradient descent) to minimize the error. This process repeats with many examples until the network achieves acceptable performance."
        },
        {
            "instruction": "What is the difference between supervised and unsupervised learning?",
            "response": "Supervised learning uses labeled data where the desired output is known, and the algorithm learns to map inputs to these correct outputs. Examples include classification and regression. Unsupervised learning works with unlabeled data, finding patterns or structures without predefined outputs. Examples include clustering and dimensionality reduction. Supervised learning requires human effort to create labels but typically achieves more precise results for specific tasks, while unsupervised learning can discover unexpected patterns but may be less directed."
        }
    ]
    
    # Create configuration for generation
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    config.sampling.temperature = 0.7
    
    # Create a generator
    generator = Generator(config)
    
    # Generate additional examples for each domain
    print("Generating additional examples...")
    general_results = generator.generate_from_seed(
        seed_examples=general_examples,
        count=10,
        method="self_instruct"
    )
    
    technical_results = generator.generate_from_seed(
        seed_examples=technical_examples,
        count=10,
        method="self_instruct"
    )
    
    print(f"Generated {len(general_results)} general examples and {len(technical_results)} technical examples")
    
    # Save the datasets
    general_output = "general_examples.jsonl"
    technical_output = "technical_examples.jsonl"
    
    general_results.save(general_output)
    technical_results.save(technical_output)
    
    print(f"Saved datasets to {general_output} and {technical_output}")
    
    # Create the metrics evaluator
    metrics = DatasetMetrics()
    
    # Evaluate and compare the datasets
    print("\n=== Dataset Quality Evaluation ===")
    
    # Basic statistics
    print("\n--- Basic Statistics ---")
    evaluate_basic_stats(general_results, "General Dataset")
    evaluate_basic_stats(technical_results, "Technical Dataset")
    
    # Diversity metrics
    print("\n--- Diversity Metrics ---")
    general_diversity = metrics.calculate_diversity(general_results)
    technical_diversity = metrics.calculate_diversity(technical_results)
    
    print(f"General Dataset Diversity Score: {general_diversity:.3f}")
    print(f"Technical Dataset Diversity Score: {technical_diversity:.3f}")
    
    # N-gram uniqueness
    print("\n--- N-gram Uniqueness ---")
    general_ngrams = metrics.calculate_ngram_uniqueness(general_results)
    technical_ngrams = metrics.calculate_ngram_uniqueness(technical_results)
    
    print("General Dataset N-gram Uniqueness:")
    for n, score in general_ngrams.items():
        print(f"  {n}-gram: {score:.3f}")
    
    print("Technical Dataset N-gram Uniqueness:")
    for n, score in technical_ngrams.items():
        print(f"  {n}-gram: {score:.3f}")
    
    # Topic distribution
    print("\n--- Topic Distribution ---")
    general_topics = metrics.extract_topics(general_results)
    technical_topics = metrics.extract_topics(technical_results)
    
    print("General Dataset Top Topics:")
    for topic, weight in general_topics[:5]:
        print(f"  {topic}: {weight:.3f}")
    
    print("Technical Dataset Top Topics:")
    for topic, weight in technical_topics[:5]:
        print(f"  {topic}: {weight:.3f}")
    
    # Visualize the results
    print("\nGenerating visualizations...")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Plot instruction and response lengths
    plot_length_distributions(general_results, technical_results, "visualizations/length_distributions.png")
    
    # Plot topic distributions
    plot_topic_distributions(general_topics, technical_topics, "visualizations/topic_distributions.png")
    
    # Readability scores
    plot_readability_scores(general_results, technical_results, "visualizations/readability_scores.png")
    
    print("\nEvaluation complete. Visualizations saved to the 'visualizations' directory.")
    
    
def evaluate_basic_stats(results, dataset_name):
    """Calculate and print basic statistics for a dataset"""
    # Convert to DataFrame for easier analysis
    df = results.to_dataframe()
    
    # Calculate lengths
    if 'instruction' in df.columns:
        instruction_lengths = df['instruction'].str.len()
        print(f"{dataset_name} Instruction Length: min={instruction_lengths.min()}, "
              f"max={instruction_lengths.max()}, mean={instruction_lengths.mean():.1f}")
    
    if 'response' in df.columns:
        response_lengths = df['response'].str.len()
        print(f"{dataset_name} Response Length: min={response_lengths.min()}, "
              f"max={response_lengths.max()}, mean={response_lengths.mean():.1f}")
    
    # Count unique instructions and responses
    if 'instruction' in df.columns:
        unique_instructions = df['instruction'].nunique()
        print(f"{dataset_name} Unique Instructions: {unique_instructions}/{len(df)} "
              f"({100 * unique_instructions / len(df):.1f}%)")


def plot_length_distributions(general_results, technical_results, output_file):
    """Plot the distribution of instruction and response lengths"""
    # Convert to DataFrames
    general_df = general_results.to_dataframe()
    technical_df = technical_results.to_dataframe()
    
    # Calculate lengths
    general_inst_len = general_df['instruction'].str.len() if 'instruction' in general_df.columns else []
    general_resp_len = general_df['response'].str.len() if 'response' in general_df.columns else []
    technical_inst_len = technical_df['instruction'].str.len() if 'instruction' in technical_df.columns else []
    technical_resp_len = technical_df['response'].str.len() if 'response' in technical_df.columns else []
    
    # Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot histograms
    if len(general_inst_len) > 0:
        axs[0, 0].hist(general_inst_len, bins=10, alpha=0.7)
        axs[0, 0].set_title('General Dataset: Instruction Length')
        axs[0, 0].set_xlabel('Length (characters)')
        axs[0, 0].set_ylabel('Count')
    
    if len(general_resp_len) > 0:
        axs[0, 1].hist(general_resp_len, bins=10, alpha=0.7)
        axs[0, 1].set_title('General Dataset: Response Length')
        axs[0, 1].set_xlabel('Length (characters)')
        axs[0, 1].set_ylabel('Count')
    
    if len(technical_inst_len) > 0:
        axs[1, 0].hist(technical_inst_len, bins=10, alpha=0.7)
        axs[1, 0].set_title('Technical Dataset: Instruction Length')
        axs[1, 0].set_xlabel('Length (characters)')
        axs[1, 0].set_ylabel('Count')
    
    if len(technical_resp_len) > 0:
        axs[1, 1].hist(technical_resp_len, bins=10, alpha=0.7)
        axs[1, 1].set_title('Technical Dataset: Response Length')
        axs[1, 1].set_xlabel('Length (characters)')
        axs[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_topic_distributions(general_topics, technical_topics, output_file):
    """Plot the distribution of topics in each dataset"""
    # Prepare data
    general_topics = general_topics[:5]  # Top 5 topics
    technical_topics = technical_topics[:5]  # Top 5 topics
    
    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot general topics
    if general_topics:
        topics, weights = zip(*general_topics)
        axs[0].barh(topics, weights, color='skyblue')
        axs[0].set_title('General Dataset: Top Topics')
        axs[0].set_xlabel('Weight')
        axs[0].invert_yaxis()  # Highest weight at the top
    
    # Plot technical topics
    if technical_topics:
        topics, weights = zip(*technical_topics)
        axs[1].barh(topics, weights, color='lightgreen')
        axs[1].set_title('Technical Dataset: Top Topics')
        axs[1].set_xlabel('Weight')
        axs[1].invert_yaxis()  # Highest weight at the top
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_readability_scores(general_results, technical_results, output_file):
    """Plot readability scores for both datasets"""
    # This is a simplified version - in a real implementation you'd use 
    # actual readability metrics like Flesch-Kincaid, SMOG, etc.
    
    # Simulate readability scores (higher = more complex)
    # In a real implementation, you would calculate these using proper metrics
    general_scores = {
        'Flesch Reading Ease': 65,
        'Flesch-Kincaid Grade': 8.5,
        'SMOG Index': 9.2,
        'Coleman-Liau Index': 10.1,
        'Automated Readability': 9.3
    }
    
    technical_scores = {
        'Flesch Reading Ease': 45,
        'Flesch-Kincaid Grade': 12.3,
        'SMOG Index': 13.1,
        'Coleman-Liau Index': 14.2,
        'Automated Readability': 13.5
    }
    
    # Create the plot
    metrics = list(general_scores.keys())
    general_values = list(general_scores.values())
    technical_values = list(technical_scores.values())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, general_values, width, label='General Dataset')
    rects2 = ax.bar(x + width/2, technical_values, width, label='Technical Dataset')
    
    ax.set_ylabel('Score')
    ax.set_title('Readability Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    
if __name__ == "__main__":
    main() 