"""
Main Generator class for synthetic data generation
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable
import importlib.util
from pathlib import Path
import time

from datagen.config import Config
from datagen.results import Results
from datagen.sampling.sampler import Sampler
from datagen.generation.self_instruct import SelfInstructGenerator
from datagen.generation.evol_instruct import EvolInstructGenerator
from datagen.quality.filter import QualityFilter
from datagen.privacy.privacy_manager import PrivacyManager

logger = logging.getLogger(__name__)


class Generator:
    """
    Main class for generating synthetic data
    """
    
    def __init__(self, config: Config):
        """
        Initialize the generator with configuration
        
        Args:
            config: Configuration for the generator
        """
        self.config = config
        self._setup_logging()
        
        # Initialize components
        self.sampler = Sampler(config.sampling)
        
        # Initialize generation methods based on config
        self.generators = {}
        if config.generation.self_instruct:
            self.generators['self_instruct'] = SelfInstructGenerator(
                config=config,
                sampler=self.sampler
            )
        if config.generation.evol_instruct:
            self.generators['evol_instruct'] = EvolInstructGenerator(
                config=config,
                sampler=self.sampler
            )
            
        # Initialize quality filter if enabled
        self.quality_filter = QualityFilter(config.quality) if config.quality.enable_filtering else None
        
        # Initialize privacy manager if enabled
        self.privacy_manager = PrivacyManager(config.privacy) if config.privacy.enable_privacy else None
        
        logger.info(f"Initialized Generator with config: {config}")
        
    def _setup_logging(self):
        """Set up logging based on configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_from_seed(
        self, 
        seed_examples: List[Dict[str, Any]], 
        count: int = 100,
        method: str = 'self_instruct',
        custom_prompt: Optional[str] = None
    ) -> Results:
        """
        Generate synthetic data from seed examples
        
        Args:
            seed_examples: List of seed examples (usually instruction-response pairs)
            count: Number of examples to generate
            method: Generation method ('self_instruct' or 'custom')
            custom_prompt: Optional custom prompt template
            
        Returns:
            Results object containing generated data
        """
        # Validate API key is available
        if self.config.generation.backend == 'openai':
            api_key = self.sampler._get_api_key('openai')
            if not api_key:
                logger.error("API key for OpenAI not found. Please set it via environment variable OPENAI_API_KEY or use generator.sampler.set_api_key('openai', 'your-key')")
                return Results([], {"error": "API key not found"})
        
        # Other validations can be added for other backends
        
        logger.info(f"Generating {count} examples using {method} from {len(seed_examples)} seed examples")
        
        # Based on method, use appropriate generator
        if method == 'self_instruct':
            if not hasattr(self, 'self_instruct_generator'):
                from datagen.generation.self_instruct import SelfInstructGenerator
                self.self_instruct_generator = SelfInstructGenerator(self.config, self.sampler)
                
            # Set custom prompt if provided
            if custom_prompt:
                original_template = self.self_instruct_generator.prompt_template
                self.self_instruct_generator.prompt_template = custom_prompt
                
            # Generate examples
            results = self.self_instruct_generator.generate(seed_examples, count)
            
            # Restore original template if changed
            if custom_prompt:
                self.self_instruct_generator.prompt_template = original_template
                
        else:
            # Custom method or error
            raise ValueError(f"Unknown generation method: {method}")
        
        # Apply quality filtering
        if self.config.quality.enable_filtering:
            results = self.quality_filter.filter(results)
        
        # Apply privacy preservation if enabled
        if self.config.privacy.enable_privacy:
            results = self.privacy_manager.process(results)
            
        return results
    
    def evolve_instructions(
        self, 
        instructions: List[str],
        rounds: int = 1,
        custom_prompt: Optional[str] = None
    ) -> Results:
        """
        Evolve instructions into more complex versions
        
        Args:
            instructions: List of instructions to evolve
            rounds: Number of evolution rounds
            custom_prompt: Optional custom prompt template for evolution
            
        Returns:
            Results object containing evolved instructions
        """
        # Validate API key is available
        if self.config.generation.backend == 'openai':
            api_key = self.sampler._get_api_key('openai')
            if not api_key:
                logger.error("API key for OpenAI not found. Please set it via environment variable OPENAI_API_KEY or use generator.sampler.set_api_key('openai', 'your-key')")
                return Results([], {"error": "API key not found"})
        
        # Other validations can be added for other backends
        
        # Set up Evol-Instruct generator
        if not hasattr(self, 'evol_instruct_generator'):
            from datagen.generation.evol_instruct import EvolInstructGenerator
            self.evol_instruct_generator = EvolInstructGenerator(self.config, self.sampler)
            
        # Set custom prompt if provided
        if custom_prompt:
            original_template = self.evol_instruct_generator.prompt_template
            self.evol_instruct_generator.prompt_template = custom_prompt
            
        # Generate evolved instructions
        results = self.evol_instruct_generator.evolve(instructions, rounds)
        
        # Restore original template if changed
        if custom_prompt:
            self.evol_instruct_generator.prompt_template = original_template
            
        # Apply quality filtering
        if self.config.quality.enable_filtering:
            results = self.quality_filter.filter(results)
            
        # Apply privacy preservation if enabled
        if self.config.privacy.enable_privacy:
            results = self.privacy_manager.process(results)
            
        return results
    
    def generate_from_template(
        self,
        template: str,
        variables: List[Dict[str, str]],
        custom_sampling_params: Optional[Dict[str, Any]] = None
    ) -> Results:
        """
        Generate data by filling a template with variables
        
        Args:
            template: Template string with placeholders for variables
            variables: List of dictionaries mapping variable names to values
            custom_sampling_params: Optional custom sampling parameters
            
        Returns:
            Results object containing generated data
        """
        logger.info(f"Generating {len(variables)} examples from template")
        start_time = time.time()
        
        data = []
        for var_set in variables:
            # Replace placeholders in template with variable values
            prompt = template
            for var_name, var_value in var_set.items():
                prompt = prompt.replace(f"{{{var_name}}}", var_value)
                
            # Generate completion using the sampler
            sampling_params = custom_sampling_params or {}
            completion = self.sampler.sample(prompt, **sampling_params)
            
            # Create result entry
            result = {
                'template': template,
                'variables': var_set,
                'prompt': prompt,
                'completion': completion
            }
            
            data.append(result)
            
        results = Results(data)
        
        # Apply quality filtering if enabled
        if self.quality_filter:
            results = self.quality_filter.filter(results)
            
        # Apply privacy measures if enabled
        if self.privacy_manager:
            results = self.privacy_manager.process(results)
            
        elapsed_time = time.time() - start_time
        logger.info(f"Generated {len(results)} examples in {elapsed_time:.2f} seconds")
        
        # Add generation metadata
        results.metadata.update({
            'generation_method': 'template',
            'template': template,
            'requested_count': len(variables),
            'generated_count': len(results),
            'generation_time': elapsed_time
        })
        
        return results
    
    def load_plugin(self, plugin_path: str) -> Any:
        """
        Load a custom plugin module
        
        Args:
            plugin_path: Path to the plugin module
            
        Returns:
            The loaded plugin module
        """
        path = Path(plugin_path)
        if not path.exists():
            raise FileNotFoundError(f"Plugin file not found: {plugin_path}")
            
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        logger.info(f"Loaded plugin from {plugin_path}")
        return module
    
    def apply_plugin_filter(self, results: Results, plugin_path: str, filter_fn_name: str) -> Results:
        """
        Apply a filter function from a plugin
        
        Args:
            results: Results to filter
            plugin_path: Path to the plugin module
            filter_fn_name: Name of the filter function in the plugin
            
        Returns:
            Filtered Results
        """
        plugin = self.load_plugin(plugin_path)
        if not hasattr(plugin, filter_fn_name):
            raise AttributeError(f"Plugin does not have a function named {filter_fn_name}")
            
        filter_fn = getattr(plugin, filter_fn_name)
        return results.filter(filter_fn) 