#!/usr/bin/env python
"""
Example of Policy Gradient (REINFORCE) for RL-guided data generation.

This example demonstrates how to use the REINFORCE algorithm to optimize 
synthetic data generation for a target model's performance. This approach
can be more efficient than random search by learning which parameters
are most effective for improving model performance.
"""

import os
import sys
import logging
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.rl_tuning import RLTuner, TORCH_AVAILABLE


def main():
    """Demonstrate policy gradient (REINFORCE) for RL tuning with a text classification task."""
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
    
    # Check for PyTorch availability
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available! ⚠️")
        print("Policy gradient (REINFORCE) requires PyTorch. Please install it using:")
        print("   pip install torch")
        print("\nContinuing with random search algorithm instead...")
    
    print("\n--- Policy Gradient (REINFORCE) for Data Generation Optimization ---")
    print("This example demonstrates how to use policy gradient methods to")
    print("optimize synthetic data generation for a text classification model.")
    
    # Create a simple validation dataset for text classification
    # In a real-world scenario, this would be your existing labeled dataset
    validation_data = [
        {"text": "I absolutely love this product, it's perfect!", "label": 1},
        {"text": "This works exactly as described, totally satisfied.", "label": 1},
        {"text": "Great quality and fast shipping, will buy again.", "label": 1},
        {"text": "I'm very impressed with the performance and quality.", "label": 1},
        {"text": "Exceeds expectations in every way, amazing product.", "label": 1},
        {"text": "This product is terrible, avoid at all costs.", "label": 0},
        {"text": "Complete waste of money, doesn't work correctly.", "label": 0},
        {"text": "Broke within a week, extremely poor quality.", "label": 0},
        {"text": "Terrible customer service and defective product.", "label": 0},
        {"text": "Not as described, had to return immediately.", "label": 0}
    ]
    
    # Create a separate test set to evaluate final performance
    test_data = [
        {"text": "I'm very happy with this purchase, worth every penny.", "label": 1},
        {"text": "Works better than expected, highly recommend.", "label": 1},
        {"text": "Excellent product that delivers on all promises.", "label": 1},
        {"text": "Don't waste your money on this junk.", "label": 0},
        {"text": "Stopped working after a few days, disappointing.", "label": 0},
        {"text": "Cheaply made and breaks easily, avoid.", "label": 0}
    ]
    
    # Create seed examples for synthetic data generation
    seed_examples = [
        {
            "instruction": "Write a positive product review for a smartphone.",
            "response": "I've been using this smartphone for the past month and it's absolutely fantastic. The camera quality is outstanding, especially in low light. Battery life is impressive, lasting a full day even with heavy use. The processor is lightning fast with no lag even when running multiple apps. The screen is bright, vibrant, and the perfect size for watching videos. The fingerprint scanner is quick and reliable. Overall, this is the best phone I've ever owned and well worth the investment."
        },
        {
            "instruction": "Write a negative product review for a laptop.",
            "response": "I regret purchasing this laptop. From day one, it's been nothing but problems. The battery drains incredibly fast, lasting barely 2 hours even with minimal usage. The keyboard has several keys that stick, making typing frustrating. It overheats constantly, even when just browsing the web. The screen has poor viewing angles and the colors look washed out. Customer support has been unhelpful and dismissive of these issues. Save yourself the headache and look elsewhere."
        }
    ]
    
    # Create a text classification target model function
    # This function will be called by the RLTuner to evaluate the synthetic data
    def target_model_fn(synthetic_examples):
        """
        Train a simple text classifier on synthetic data and evaluate on validation set.
        
        Args:
            synthetic_examples: List of dictionaries with 'instruction' and 'response' fields
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract text and create labels
        texts = []
        labels = []
        
        for example in synthetic_examples:
            text = example.get('response', '')
            
            # Determine label based on instruction content
            instruction = example.get('instruction', '').lower()
            if 'positive' in instruction:
                label = 1
            elif 'negative' in instruction:
                label = 0
            else:
                # Skip examples we can't clearly label
                continue
            
            texts.append(text)
            labels.append(label)
        
        # Skip if not enough examples
        if len(texts) < 4 or len(set(labels)) < 2:
            print("Not enough examples or class diversity for training")
            return {"accuracy": 0, "f1": 0, "examples_used": len(texts)}
        
        # Create feature vectors with TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000)
        X_train = vectorizer.fit_transform(texts)
        y_train = np.array(labels)
        
        # Extract validation features
        val_texts = [item['text'] for item in validation_data]
        val_labels = [item['label'] for item in validation_data]
        X_val = vectorizer.transform(val_texts)
        y_val = np.array(val_labels)
        
        # Train a simple logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)
        f1 = f1_score(y_val, val_predictions)
        
        # Evaluate on test set
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        X_test = vectorizer.transform(test_texts)
        y_test = np.array(test_labels)
        
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions)
        
        # Return metrics
        return {
            "accuracy": accuracy,
            "f1": f1,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "examples_used": len(texts),
            "class_distribution": f"{sum(y_train == 1)}/{sum(y_train == 0)}"
        }
    
    # Create a configuration for RL tuning
    config = Config()
    config.generation.model_name = "gpt-3.5-turbo"
    config.generation.backend = "openai"
    
    # Configure RL tuning settings
    config.rl_tuning.enable_rl_tuning = True
    config.rl_tuning.num_iterations = 8  # More iterations for policy learning
    config.rl_tuning.batch_size = 6  # Small for demonstration
    
    # Set algorithm to REINFORCE (fall back to random_search if PyTorch not available)
    if TORCH_AVAILABLE:
        config.rl_tuning.rl_algorithm = "reinforce"
        # Configure policy gradient specific parameters
        config.rl_tuning.policy_hidden_dim = 32
        config.rl_tuning.gamma = 0.95
        config.rl_tuning.learning_rate = 0.005
        config.rl_tuning.normalize_rewards = True
    else:
        config.rl_tuning.rl_algorithm = "random_search"
    
    config.rl_tuning.reward_metric = "f1"  # Optimize for F1 score
    
    # Create the RLTuner
    tuner = RLTuner(
        config=config,
        target_model=target_model_fn,
        validation_dataset=validation_data
    )
    
    # Run the RL tuning process
    print("\nStarting policy gradient optimization. This may take a while...")
    results = tuner.train(seed_examples=seed_examples)
    
    # Print results
    print("\n--- Optimization Results ---")
    print(f"Algorithm used: {results['algorithm']}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Best F1 score: {results['best_reward']:.4f}")
    
    # Create a visualization of the training progress
    plot_training_progress(results)
    
    # Generate final dataset with best parameters
    print("\nGenerating final dataset with best parameters...")
    final_dataset = tuner.generate_with_best_params(count=20, seed_examples=seed_examples)
    
    # Evaluate the final dataset
    final_metrics = target_model_fn(final_dataset.data)
    
    print("\n--- Final Evaluation ---")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation F1 Score: {final_metrics['f1']:.4f}")
    print(f"Test Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"Test F1 Score: {final_metrics['test_f1']:.4f}")
    print(f"Examples used: {final_metrics['examples_used']}")
    print(f"Class distribution: {final_metrics['class_distribution']}")
    
    # Save the best dataset
    output_file = "policy_gradient_optimized_data.jsonl"
    final_dataset.save(output_file)
    print(f"\nSaved optimized dataset to {output_file}")
    
    # Save the tuner state
    tuner.save("policy_gradient_tuner_state.json")
    print("Saved tuner state to policy_gradient_tuner_state.json")
    
    # Compare policy gradient (or random search) with baseline
    compare_with_baseline(tuner, target_model_fn, seed_examples)
    
    # Display a sample of the generated data
    print("\nSample of generated data with best parameters:")
    for i, example in enumerate(final_dataset.data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example.get('instruction', '')}")
        print(f"Response: {example.get('response', '')[:150]}...")


def plot_training_progress(results):
    """
    Plot the training progress with rewards over iterations.
    
    Args:
        results: Dictionary with training results
    """
    try:
        # Extract data
        iterations = [entry['iteration'] for entry in results['history']]
        rewards = [entry['reward'] for entry in results['history']]
        temperatures = [entry['params']['temperature'] for entry in results['history']]
        top_ps = [entry['params']['top_p'] for entry in results['history']]
        
        # Create figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot rewards
        ax1.plot(iterations, rewards, 'b-', marker='o')
        ax1.axhline(y=results['best_reward'], color='r', linestyle='--', alpha=0.7, 
                   label=f"Best Reward: {results['best_reward']:.4f}")
        ax1.set_ylabel(f"{results['reward_metric'].upper()} Score")
        ax1.set_title(f"RL Tuning Progress with {results['algorithm'].upper()}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot parameters
        ax2.plot(iterations, temperatures, 'g-', marker='s', label='Temperature')
        ax2.plot(iterations, top_ps, 'm-', marker='^', label='Top-p')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Annotate best iteration
        best_iteration = iterations[rewards.index(results['best_reward'])]
        ax1.annotate(f'Best', xy=(best_iteration, results['best_reward']), 
                    xytext=(best_iteration, results['best_reward']*0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.tight_layout()
        plt.savefig('rl_tuning_progress.png')
        plt.close()
        
        print(f"\nTraining progress plot saved to 'rl_tuning_progress.png'")
    except Exception as e:
        print(f"Error creating plot: {e}")


def compare_with_baseline(tuner, target_model_fn, seed_examples):
    """
    Compare RL-optimized generation with baseline (default parameters).
    
    Args:
        tuner: Trained RLTuner instance
        target_model_fn: Target model evaluation function
        seed_examples: Seed examples for generation
    """
    print("\n--- Comparison with Baseline ---")
    
    # Create a baseline generator with default parameters
    baseline_config = Config()
    baseline_config.generation.model_name = tuner.config.generation.model_name
    baseline_config.generation.backend = tuner.config.generation.backend
    baseline_generator = Generator(baseline_config)
    
    # Generate data with baseline parameters
    print("Generating data with baseline parameters...")
    baseline_results = baseline_generator.generate_from_seed(
        seed_examples=seed_examples,
        count=20,
        method="self_instruct"
    )
    
    # Evaluate baseline
    baseline_metrics = target_model_fn(baseline_results.data)
    
    # Get metrics from the optimized data
    optimized_metrics = target_model_fn(
        tuner.generate_with_best_params(count=20, seed_examples=seed_examples).data
    )
    
    # Print comparison
    print("\nBaseline parameters:")
    print(f"  Temperature: {baseline_config.sampling.temperature}")
    print(f"  Top-p: {baseline_config.sampling.top_p}")
    print(f"  Generation method: self_instruct")
    
    print("\nOptimized parameters:")
    print(f"  Temperature: {tuner.best_params['temperature']}")
    print(f"  Top-p: {tuner.best_params['top_p']}")
    print(f"  Generation method: {tuner.best_params['generation_method']}")
    
    print("\nMetrics comparison:")
    print(f"  Baseline F1 score:     {baseline_metrics['f1']:.4f}")
    print(f"  Optimized F1 score:    {optimized_metrics['f1']:.4f}")
    print(f"  Improvement:           {optimized_metrics['f1'] - baseline_metrics['f1']:.4f} " +
          f"({(optimized_metrics['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100:.1f}%)")


if __name__ == "__main__":
    main() 