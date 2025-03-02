"""
Evol-Instruct generation method implementation.

Based on the paper: "Automatic Instruction Evol-Instruct via Chain-of-Thought"
https://arxiv.org/abs/2305.12475
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

# Default prompt template for Evol-Instruct generation
DEFAULT_EVOL_PROMPT_TEMPLATE = """
You are tasked with making an instruction more complex, specific, and challenging. 
Given the instruction below, please create a more advanced version that:
1. Adds more specific details and constraints
2. Increases the complexity and depth required
3. Makes the instruction more challenging while still being reasonable
4. Maintains the core intent of the original instruction

Original instruction:
{instruction}

Now, create a more complex and specific version of this instruction:
"""


class EvolInstructGenerator:
    """
    Implementation of the Evol-Instruct method for evolving instructions
    into more complex and challenging versions
    """
    
    def __init__(self, config: Config, sampler: Sampler):
        """
        Initialize the Evol-Instruct generator
        
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
            template_path = Path(self.config.generation.prompt_templates_dir) / "evol_instruct.txt"
            if template_path.exists():
                with open(template_path, "r") as f:
                    self.prompt_template = f.read()
            else:
                logger.warning(f"Prompt template file not found at {template_path}. Using default.")
                self.prompt_template = DEFAULT_EVOL_PROMPT_TEMPLATE
        else:
            self.prompt_template = DEFAULT_EVOL_PROMPT_TEMPLATE
            
    def evolve_single_instruction(
        self, 
        instruction: str, 
        rounds: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Evolve a single instruction through multiple rounds
        
        Args:
            instruction: Original instruction
            rounds: Number of evolution rounds
            
        Returns:
            List of evolved instructions with metadata
        """
        results = []
        current_instruction = instruction
        
        for round_idx in range(rounds):
            prompt = self.prompt_template.format(instruction=current_instruction)
            
            # Add any custom variables from config
            for var_name, var_value in self.config.generation.custom_prompt_variables.items():
                prompt = prompt.replace(f"{{{var_name}}}", var_value)
                
            try:
                evolved_instruction = self.sampler.sample(
                    prompt=prompt,
                    backend=self.config.generation.backend,
                    model=self.config.generation.model_name
                )
                
                # Clean up the generated instruction
                evolved_instruction = evolved_instruction.strip()
                
                # Save the result
                result = {
                    "original_instruction": instruction if round_idx == 0 else results[-1]["evolved_instruction"],
                    "evolved_instruction": evolved_instruction,
                    "evolution_round": round_idx + 1,
                    "source": "evol_instruct"
                }
                
                results.append(result)
                
                # Update for next round
                current_instruction = evolved_instruction
                
            except Exception as e:
                logger.error(f"Error in evolution round {round_idx + 1}: {e}")
                break
                
        return results
    
    def evolve(self, instructions: List[str], rounds: int = 1) -> Results:
        """
        Evolve a list of instructions
        
        Args:
            instructions: List of instructions to evolve
            rounds: Number of evolution rounds
            
        Returns:
            Results object containing evolved instructions
        """
        all_results = []
        metadata = {
            "evol_instruct": {
                "original_instructions_count": len(instructions),
                "evolution_rounds": rounds,
            }
        }
        
        logger.info(f"Evolving {len(instructions)} instructions through {rounds} rounds")
        start_time = time.time()
        
        for idx, instruction in enumerate(instructions):
            logger.info(f"Evolving instruction {idx + 1}/{len(instructions)}")
            
            evolved = self.evolve_single_instruction(instruction, rounds)
            all_results.extend(evolved)
            
        elapsed_time = time.time() - start_time
        logger.info(f"Evolved {len(instructions)} instructions into {len(all_results)} in {elapsed_time:.2f} seconds")
        
        # Add metadata
        metadata["evol_instruct"]["evolution_time"] = elapsed_time
        metadata["evol_instruct"]["evolved_instructions_count"] = len(all_results)
        
        return Results(all_results, metadata)
    
    def generate(self, seed_examples: List[Dict[str, Any]], count: int = 100) -> Results:
        """
        Generate evolved instructions from seed examples
        
        This method extracts instructions from seed examples and evolves them.
        
        Args:
            seed_examples: Seed examples with instructions
            count: Number of examples to generate
            
        Returns:
            Results object containing evolved instructions
        """
        # Extract instructions from seed examples
        instructions = []
        for ex in seed_examples:
            if "instruction" in ex:
                instructions.append(ex["instruction"])
                
        if not instructions:
            raise ValueError("No instructions found in seed examples")
            
        # Sample instructions if we have more than we need
        if len(instructions) > count:
            instructions = random.sample(instructions, count)
            
        # Calculate rounds needed to reach count
        rounds = max(1, (count + len(instructions) - 1) // len(instructions))
        
        # Evolve instructions
        results = self.evolve(instructions, rounds)
        
        # Limit to requested count
        if len(results.data) > count:
            results.data = results.data[:count]
            
        return results 