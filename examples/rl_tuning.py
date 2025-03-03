#!/usr/bin/env python
"""
Example of RL-guided data generation to optimize synthetic data for a target model.

This example demonstrates how to use the RLTuner to generate synthetic data
that improves a target model's performance on a specific task.
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

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.rl_tuning import RLTuner


def main():
    """Demonstrate RL tuning with a simple text classification example."""
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
    
    print("\n--- RL-Guided Data Generation Example ---")
    print("This example demonstrates how to use RL to optimize synthetic data generation")
    print("for improving a text classification model.")
    
    # Create a simple validation dataset for sentiment classification
    # This would typically be provided by the user
    validation_data = [
        {"text": "I love this product, it's amazing!", "label": 1},
        {"text": "This works exactly as described, very happy.", "label": 1},
        {"text": "Great quality and fast shipping.", "label": 1},
        {"text": "I'm impressed with the performance.", "label": 1},
        {"text": "Exceeds expectations in every way.", "label": 1},
        {"text": "This product is terrible, don't buy it.", "label": 0},
        {"text": "Waste of money, doesn't work at all.", "label": 0},
        {"text": "Broke after two days, very disappointing.", "label": 0},
        {"text": "Poor quality and bad customer service.", "label": 0},
        {"text": "Not as described, had to return it.", "label": 0}
    ]
    
    # Create a separate small test set to validate final performance
    test_data = [
        {"text": "I'm very satisfied with my purchase.", "label": 1},
        {"text": "Works perfectly, would buy again.", "label": 1},
        {"text": "This product changed my life for the better.", "label": 1},
        {"text": "Doesn't work properly, avoid this product.", "label": 0},
        {"text": "Save your money and look elsewhere.", "label": 0},
        {"text": "The worst purchase I've ever made.", "label": 0}
    ]
    
    # Create seed examples for synthetic data generation
    seed_examples = [
        {
            "instruction": "Write a positive product review.",
            "response": "I absolutely love this camera! The image quality is superb, and the auto-focus feature works flawlessly. Battery life is excellent, lasting all day on a single charge. The interface is intuitive and easy to navigate. It's lightweight yet feels sturdy and well-built. Definitely worth the investment if you're serious about photography."
        },
        {
            "instruction": "Write a negative product review.",
            "response": "I'm extremely disappointed with this laptop. It overheats constantly, even when just browsing the web. The battery drains completely in less than 2 hours. The keyboard has several keys that stick, and the trackpad is unresponsive at times. Customer service was unhelpful when I contacted them about these issues. Would not recommend to anyone."
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
        # For this example, we assume 'Write a positive review' creates positive examples
        # and 'Write a negative review' creates negative examples
        texts = []
        labels = []
        
        for example in synthetic_examples:
            # Use the response as the text to classify
            text = example.get('response', '')
            
            # Determine the label based on the instruction
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
        
        # Evaluate on the separate test set for a more robust measurement
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
    config.rl_tuning.num_iterations = 5  # Small for demonstration
    config.rl_tuning.batch_size = 6  # Small for demonstration
    config.rl_tuning.rl_algorithm = "random_search"
    config.rl_tuning.reward_metric = "f1"  # Optimize for F1 score
    
    # Create the RLTuner
    tuner = RLTuner(
        config=config,
        target_model=target_model_fn,
        validation_dataset=validation_data
    )
    
    # Run the RL tuning process
    print("\nStarting RL tuning process. This may take a while...")
    results = tuner.train(seed_examples=seed_examples)
    
    # Print results
    print("\n--- RL Tuning Results ---")
    print(f"Best parameters: {results['best_params']}")
    print(f"Best F1 score: {results['best_reward']:.4f}")
    
    # Print history summary
    print("\nIteration history:")
    for entry in results['history']:
        print(f"Iteration {entry['iteration']}: "
              f"params={entry['params']}, "
              f"f1={entry['metrics']['f1']:.4f}, "
              f"examples={entry['examples_count']}")
    
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
    output_file = "rl_optimized_data.jsonl"
    final_dataset.save(output_file)
    print(f"\nSaved RL-optimized dataset to {output_file}")
    
    # Save the tuner state for later use
    tuner.save("rl_tuner_state.json")
    print("Saved RLTuner state to rl_tuner_state.json")
    
    # Display a sample of the generated data
    print("\nSample of generated data with best parameters:")
    for i, example in enumerate(final_dataset.data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example.get('instruction', '')}")
        print(f"Response: {example.get('response', '')[:150]}...")


if __name__ == "__main__":
    main() 