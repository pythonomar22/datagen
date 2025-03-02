"""
Self-Instruct generation method implementation.

Based on the paper: "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
https://arxiv.org/abs/2212.10560
"""

import os
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re

from datagen.config import Config
from datagen.sampling.sampler import Sampler
from datagen.results import Results

logger = logging.getLogger(__name__)

# Default prompt template for Self-Instruct generation
DEFAULT_PROMPT_TEMPLATE = """
You are tasked with generating diverse, high-quality instructions and their corresponding responses for training AI assistants. 
Please create new examples that are different from the examples below.

Here are some examples of instruction-response pairs:

{seed_examples}

Now, create {count} new instruction-response pairs that are diverse, creative, and helpful.
Each new pair should be formatted as:
Instruction: <instruction>
Response: <response>

Make sure to include a wide range of topics, difficulty levels, and response types.
"""


class SelfInstructGenerator:
    """
    Implementation of the Self-Instruct method for generating synthetic instruction-response pairs
    """
    
    def __init__(self, config: Config, sampler: Sampler):
        """
        Initialize the Self-Instruct generator
        
        Args:
            config: Configuration object
            sampler: Sampler for text generation
        """
        self.config = config
        self.sampler = sampler
        
        # Load or set default prompt template
        self.prompt_template = None
        self._load_prompt_template()
        
    def _load_prompt_template(self):
        """Load prompt template from configuration or use default"""
        if self.config.generation.prompt_template:
            self.prompt_template = self.config.generation.prompt_template
        elif self.config.generation.prompt_templates_dir:
            template_path = Path(self.config.generation.prompt_templates_dir) / "self_instruct.txt"
            if template_path.exists():
                with open(template_path, "r") as f:
                    self.prompt_template = f.read()
            else:
                logger.warning(f"Prompt template file not found at {template_path}. Using default.")
                self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        else:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATE
            
    def _format_seed_examples(self, seed_examples: List[Dict[str, Any]]) -> str:
        """Format seed examples for inclusion in the prompt"""
        formatted_examples = []
        
        for ex in seed_examples:
            if "instruction" in ex and "response" in ex:
                formatted = f"Instruction: {ex['instruction']}\nResponse: {ex['response']}"
                formatted_examples.append(formatted)
                
        return "\n\n".join(formatted_examples)
        
    def _parse_generation(self, text: str) -> List[Dict[str, str]]:
        """Parse the generated text into instruction-response pairs"""
        # Regex pattern to match instruction-response pairs
        pattern = r"Instruction:\s*(.*?)\s*Response:\s*(.*?)(?=Instruction:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        
        results = []
        for instr, resp in matches:
            # Clean and validate the pair
            instruction = instr.strip()
            response = resp.strip()
            
            if instruction and response:
                results.append({
                    "instruction": instruction,
                    "response": response
                })
                
        return results
        
    def generate(
        self, 
        seed_examples: List[Dict[str, Any]], 
        count: int = 100,
        batch_size: int = 10
    ) -> Results:
        """
        Generate synthetic instruction-response pairs using Self-Instruct
        
        Args:
            seed_examples: Seed examples to use as demonstrations
            count: Total number of examples to generate
            batch_size: Number of examples to generate per batch
            
        Returns:
            Results object containing generated data
        """
        if not seed_examples:
            raise ValueError("At least one seed example is required for Self-Instruct generation")
            
        # Ensure we have all required fields in seed examples
        for ex in seed_examples:
            if "instruction" not in ex or "response" not in ex:
                raise ValueError("Seed examples must contain 'instruction' and 'response' fields")
                
        # Calculate batches needed
        batch_count = (count + batch_size - 1) // batch_size  # Ceiling division
        instances_per_batch = min(batch_size, count)
        
        all_results = []
        metadata = {
            "self_instruct": {
                "seed_examples_count": len(seed_examples),
                "batches": batch_count,
            }
        }
        
        formatted_seeds = self._format_seed_examples(seed_examples)
        
        logger.info(f"Generating {count} examples in {batch_count} batches")
        start_time = time.time()
        
        for batch_idx in range(batch_count):
            # Prepare prompt for this batch
            prompt = self.prompt_template.format(
                seed_examples=formatted_seeds,
                count=instances_per_batch
            )
            
            # Add any custom variables from config
            for var_name, var_value in self.config.generation.custom_prompt_variables.items():
                prompt = prompt.replace(f"{{{var_name}}}", var_value)
            
            # Generate text
            try:
                generated_text = self.sampler.sample(
                    prompt=prompt,
                    backend=self.config.generation.backend,
                    model=self.config.generation.model_name
                )
                
                # Parse the generated text
                pairs = self._parse_generation(generated_text)
                
                logger.info(f"Batch {batch_idx + 1}/{batch_count}: Generated {len(pairs)} instruction-response pairs")
                
                # Add source information
                for pair in pairs:
                    pair["source"] = "self_instruct"
                    pair["batch"] = batch_idx
                    
                all_results.extend(pairs)
                
            except Exception as e:
                logger.error(f"Error generating batch {batch_idx + 1}: {e}")
                
            # Early stopping if we have enough examples
            if len(all_results) >= count:
                break
                
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(all_results)} examples in {elapsed_time:.2f} seconds")
        
        # Create Results object
        metadata["self_instruct"]["generation_time"] = elapsed_time
        
        # Limit to the requested count
        return Results(all_results[:count], metadata) 