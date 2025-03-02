"""
Quality filtering for synthetic data
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable, Set
import time
from collections import Counter
import difflib

from datagen.config import QualityConfig
from datagen.results import Results

logger = logging.getLogger(__name__)


class QualityFilter:
    """Quality filtering for synthetic data"""
    
    def __init__(self, config: QualityConfig):
        """
        Initialize the quality filter
        
        Args:
            config: Quality filter configuration
        """
        self.config = config
        self.filters = []
        
        # Register default filters based on config
        self._register_default_filters()
        
    def _register_default_filters(self):
        """Register default filters based on configuration"""
        # Basic length filters
        if self.config.min_length > 0:
            self.register_filter(self.filter_min_length)
            
        if self.config.max_length < float('inf'):
            self.register_filter(self.filter_max_length)
            
        # Instruction-specific filters
        if self.config.min_instruction_length > 0:
            self.register_filter(self.filter_instruction_min_length)
            
        if self.config.min_response_length > 0:
            self.register_filter(self.filter_response_min_length)
            
        # Duplicate detection
        if self.config.similarity_threshold > 0:
            self.register_filter(self.filter_duplicate_content)
            
        # Other filters can be added here based on config parameters
            
    def register_filter(self, filter_fn: Callable[[Dict[str, Any]], bool]):
        """
        Register a filter function
        
        Args:
            filter_fn: Function that takes an example and returns True to keep it
        """
        self.filters.append(filter_fn)
        logger.debug(f"Registered filter: {filter_fn.__name__}")
    
    def filter(self, results: Results) -> Results:
        """
        Apply all registered filters to the results
        
        Args:
            results: Results to filter
            
        Returns:
            Filtered Results
        """
        start_time = time.time()
        original_count = len(results)
        
        # Check if there are any examples to filter
        if original_count == 0:
            logger.warning("No examples to filter")
            return results
        
        # Apply each filter in sequence
        for filter_fn in self.filters:
            filter_name = filter_fn.__name__
            logger.info(f"Applying filter: {filter_name}")
            
            # Apply the filter
            results = results.filter(filter_fn)
            
            # Log results
            removed = original_count - len(results)
            percentage = removed / original_count if original_count > 0 else 0
            logger.info(f"Filter {filter_name} removed {removed} examples ({percentage:.1%})")
            
        elapsed_time = time.time() - start_time
        retention_rate = len(results) / original_count if original_count > 0 else 0
        logger.info(f"Filtering completed in {elapsed_time:.2f}s. "
                  f"Kept {len(results)}/{original_count} examples ({retention_rate:.1%})")
        
        # Add filtering metadata
        if 'filtering' not in results.metadata:
            results.metadata['filtering'] = {}
            
        results.metadata['filtering'].update({
            'original_count': original_count,
            'filtered_count': len(results),
            'removed_count': original_count - len(results),
            'filtering_time': elapsed_time
        })
        
        return results
    
    # ------------------------
    # Default Filter Functions
    # ------------------------
    
    def filter_min_length(self, example: Dict[str, Any]) -> bool:
        """Filter examples with insufficient length"""
        # Check each text field
        for value in example.values():
            if isinstance(value, str) and len(value) < self.config.min_length:
                return False
        return True
    
    def filter_max_length(self, example: Dict[str, Any]) -> bool:
        """Filter examples that are too long"""
        # Check each text field
        for value in example.values():
            if isinstance(value, str) and len(value) > self.config.max_length:
                return False
        return True
    
    def filter_instruction_min_length(self, example: Dict[str, Any]) -> bool:
        """Filter examples with instruction field that's too short"""
        if 'instruction' in example:
            return len(example['instruction']) >= self.config.min_instruction_length
        elif 'evolved_instruction' in example:
            return len(example['evolved_instruction']) >= self.config.min_instruction_length
        return True  # No instruction field to check
    
    def filter_response_min_length(self, example: Dict[str, Any]) -> bool:
        """Filter examples with response field that's too short"""
        if 'response' in example:
            return len(example['response']) >= self.config.min_response_length
        return True  # No response field to check
        
    def filter_duplicate_content(self, example: Dict[str, Any]) -> bool:
        """
        Filter duplicative content (approximate matching)
        
        This is a stateful filter that needs to track previous examples.
        It uses a simple class attribute to track seen examples.
        """
        # Initialize class storage for seen content if not present
        if not hasattr(self, '_seen_content'):
            self._seen_content = set()
            
        # For instruction-response pairs, check both together
        if 'instruction' in example and 'response' in example:
            content = f"{example['instruction']} {example['response']}"
        # For evolved instructions
        elif 'evolved_instruction' in example:
            content = example['evolved_instruction']
        # For other text content
        else:
            # Concatenate all string values
            content = " ".join(str(v) for v in example.values() if isinstance(v, str))
            
        # Check similarity with seen content
        for seen in self._seen_content:
            similarity = self._calculate_similarity(content, seen)
            if similarity > self.config.similarity_threshold:
                return False
                
        # Add to seen content
        self._seen_content.add(content)
        return True
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def filter_perplexity(self, example: Dict[str, Any]) -> bool:
        """
        Filter based on perplexity (requires a language model)
        
        This filter requires a perplexity calculator that would be initialized
        elsewhere and provided to the filter.
        """
        if not hasattr(self, '_perplexity_calculator'):
            logger.warning("Perplexity filter used but no calculator available")
            return True
            
        if self.config.perplexity_threshold is None:
            return True
            
        # Calculate perplexity for relevant fields
        if 'instruction' in example:
            perplexity = self._perplexity_calculator(example['instruction'])
            if perplexity > self.config.perplexity_threshold:
                return False
                
        if 'response' in example:
            perplexity = self._perplexity_calculator(example['response'])
            if perplexity > self.config.perplexity_threshold:
                return False
                
        if 'evolved_instruction' in example:
            perplexity = self._perplexity_calculator(example['evolved_instruction'])
            if perplexity > self.config.perplexity_threshold:
                return False
                
        return True
    
    def set_perplexity_calculator(self, calculator_fn: Callable[[str], float]):
        """
        Set the perplexity calculator function
        
        Args:
            calculator_fn: Function that calculates perplexity of a text
        """
        self._perplexity_calculator = calculator_fn
        # Register the perplexity filter if we have a threshold
        if self.config.perplexity_threshold is not None:
            self.register_filter(self.filter_perplexity) 