#!/usr/bin/env python
"""
Example of integrating DataGen with ML training pipelines (PyTorch)
"""

import os
import sys
import logging
import tempfile
import json
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    GPT2Config,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datagen import Generator, Config, Results
from datagen.pipeline.io import DataLoader, DataExporter


class SyntheticDataset(Dataset):
    """Dataset for training with synthetic data"""
    
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the example as a prompt-response pair
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        
        text = f"Instruction: {instruction}\nResponse: {response}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Return as a dictionary
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }


def train_model_on_synthetic_data(train_dataset, val_dataset, output_dir, model_name="gpt2", epochs=3):
    """Train a small LM on synthetic data"""
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a smaller model configuration for demo purposes
    config = GPT2Config.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        n_positions=512,
        n_ctx=512,
        n_layer=6,
        n_head=8
    )
    
    model = GPT2LMHeadModel(config)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        no_cuda=not torch.cuda.is_available(),
        seed=42,
        load_best_model_at_end=True,
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Training model on synthetic data...")
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return model, tokenizer


def generate_text_with_model(model, tokenizer, prompt, max_length=100):
    """Generate text using the fine-tuned model"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def main():
    """Demonstrate ML integration with DataGen"""
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
    
    print("This example demonstrates how to integrate DataGen with PyTorch for model training.")
    print("Note: This will train a small language model on synthetic data.")
    print("For demonstration purposes, we'll use a very small model and limited training.")
    
    # Create a temporary working directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nCreated temporary directory for training files: {temp_dir}")
    
    try:
        # Step 1: Generate synthetic training data
        print("\n--- Step 1: Generating synthetic training data ---")
        
        config = Config()
        config.generation.model_name = "gpt-3.5-turbo"
        config.generation.backend = "openai"
        config.sampling.temperature = 0.7
        
        generator = Generator(config)
        
        # Define seed examples for a Q&A dataset
        seed_examples = [
            {
                "instruction": "What are black holes?",
                "response": "Black holes are regions of spacetime where gravity is so strong that nothing, including light or other electromagnetic waves, can escape from it. They form when massive stars collapse at the end of their life cycle. At the center of a black hole is a singularity, a point of infinite density where our current physics laws break down."
            },
            {
                "instruction": "Explain how electric cars work.",
                "response": "Electric cars run on electricity stored in rechargeable batteries, which power an electric motor that drives the wheels. Unlike conventional vehicles with internal combustion engines, electric cars don't burn fuel, have no exhaust emissions, and typically have fewer moving parts. When the batteries deplete, they can be recharged by plugging into a charging station or outlet. The regenerative braking system helps recharge the battery while driving by converting kinetic energy back to electrical energy."
            }
        ]
        
        # Generate synthetic training data
        print("\nGenerating training examples...")
        training_results = generator.generate_from_seed(
            seed_examples=seed_examples,
            count=20,  # Small count for demonstration
            method="self_instruct"
        )
        
        print(f"Generated {len(training_results)} training examples")
        
        # Generate validation data separately for better evaluation
        print("\nGenerating validation examples...")
        validation_results = generator.generate_from_seed(
            seed_examples=seed_examples,
            count=5,  # Small count for demonstration
            method="self_instruct"
        )
        
        print(f"Generated {len(validation_results)} validation examples")
        
        # Save the datasets
        train_path = os.path.join(temp_dir, "train.jsonl")
        val_path = os.path.join(temp_dir, "val.jsonl")
        
        training_results.save(train_path)
        validation_results.save(val_path)
        
        print(f"Saved training data to: {train_path}")
        print(f"Saved validation data to: {val_path}")
        
        # Step 2: Prepare data for PyTorch training
        print("\n--- Step 2: Preparing data for PyTorch training ---")
        
        # Initialize tokenizer for data processing
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create PyTorch datasets
        train_dataset = SyntheticDataset(training_results.data, tokenizer)
        val_dataset = SyntheticDataset(validation_results.data, tokenizer)
        
        print(f"Created training dataset with {len(train_dataset)} examples")
        print(f"Created validation dataset with {len(val_dataset)} examples")
        
        # Step 3: Train a small model on the synthetic data
        print("\n--- Step 3: Training a model on synthetic data ---")
        print("Note: Training is limited for demonstration purposes. In a real scenario, you would use more data and longer training.")
        
        # Set output directory
        output_dir = os.path.join(temp_dir, "model_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # In a real example, we would train the model
        # For demo purposes, we'll just describe the process
        print("\nIn a full implementation, we would now train the model with code like:")
        print("model, tokenizer = train_model_on_synthetic_data(train_dataset, val_dataset, output_dir)")
        print("\nSince training even a small model requires significant compute resources,")
        print("this example stops short of actual training but demonstrates the full code flow.")
        
        # Step 4: Demonstrate model inference
        print("\n--- Step 4: Model inference demonstration ---")
        print("After training, you would use the model for inference like this:")
        
        # Example prompts for inference
        test_prompts = [
            "Instruction: How do transistors work?",
            "Instruction: Explain the water cycle on Earth."
        ]
        
        print("\nExample inference (simulated):")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print("Response: [This would show the model's generated response if we had trained it]")
        
        print("\nTo integrate this with a real ML workflow, you would:")
        print("1. Generate larger synthetic datasets with DataGen")
        print("2. Format the data appropriately for your model architecture")
        print("3. Train models using your preferred ML framework (PyTorch, TensorFlow, etc.)")
        print("4. Evaluate performance and iterate on data generation parameters")
    
    finally:
        # Cleanup (commented out to allow inspection)
        print(f"\nTemporary directory with files: {temp_dir}")
        print("Note: The temporary directory was NOT deleted so you can examine the files.")
        print("Please manually delete it when you're done.")
    
    
if __name__ == "__main__":
    main() 