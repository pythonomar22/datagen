"""
Privacy preservation functionality for synthetic data
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
import time
import numpy as np
import random

from datagen.config import PrivacyConfig
from datagen.results import Results

logger = logging.getLogger(__name__)


class PrivacyManager:
    """Manager for privacy preservation in generated data"""
    
    def __init__(self, config: PrivacyConfig):
        """
        Initialize the privacy manager
        
        Args:
            config: Privacy configuration
        """
        self.config = config
        
    def process(self, results: Results) -> Results:
        """
        Apply privacy measures to the results
        
        Args:
            results: Results to process
            
        Returns:
            Processed Results
        """
        if not self.config.enable_privacy:
            logger.info("Privacy preservation disabled, skipping")
            return results
            
        start_time = time.time()
        logger.info(f"Applying privacy preservation to {len(results)} examples")
        
        processed_results = results
        
        # Apply each privacy measure in sequence
        
        # Apply differential privacy if enabled
        if self.config.differential_privacy:
            processed_results = self._apply_differential_privacy(processed_results)
            
        # Apply content filtering
        if self.config.enable_content_filtering:
            processed_results = self._apply_content_filtering(processed_results)
            
        elapsed_time = time.time() - start_time
        logger.info(f"Privacy processing completed in {elapsed_time:.2f}s")
        
        # Add privacy metadata
        processed_results.metadata['privacy'] = {
            'differential_privacy': self.config.differential_privacy,
            'dp_epsilon': self.config.dp_epsilon if self.config.differential_privacy else None,
            'dp_delta': self.config.dp_delta if self.config.differential_privacy else None,
            'content_filtering': self.config.enable_content_filtering,
            'processing_time': elapsed_time
        }
        
        return processed_results
    
    def _apply_differential_privacy(self, results: Results) -> Results:
        """
        Apply differential privacy to the results
        
        This is a simplified implementation of differential privacy by adding noise
        to text embeddings or adding random perturbations to text.
        
        In a production setting, this would be replaced with a more sophisticated DP algorithm.
        
        Args:
            results: Results to process
            
        Returns:
            Processed Results
        """
        logger.info(f"Applying differential privacy (epsilon={self.config.dp_epsilon}, delta={self.config.dp_delta})")
        
        # Simplified DP implementation: randomly perturb some words
        def apply_dp_to_text(text: str) -> str:
            # This is a naive implementation and should be replaced with a proper DP algorithm
            words = text.split()
            
            # Determine perturbation level based on epsilon (lower epsilon = more privacy = more noise)
            # This is a very simplified approach
            perturbation_prob = min(0.05, 1.0 / self.config.dp_epsilon)
            
            for i in range(len(words)):
                # Randomly perturb words with probability based on epsilon
                if random.random() < perturbation_prob:
                    options = [
                        lambda w: w,  # keep original
                        lambda w: w.lower(),  # lowercase
                        lambda w: w.upper() if len(w) <= 3 else w,  # uppercase short words
                        lambda w: w + " " if len(w) > 4 else w,  # add space after
                    ]
                    words[i] = random.choice(options)(words[i])
                    
            return " ".join(words)
            
        # Process each example
        def dp_process(example: Dict[str, Any]) -> Dict[str, Any]:
            result = example.copy()
            
            # Apply DP to relevant text fields
            if 'instruction' in result:
                result['instruction'] = apply_dp_to_text(result['instruction'])
                
            if 'response' in result:
                result['response'] = apply_dp_to_text(result['response'])
                
            if 'evolved_instruction' in result:
                result['evolved_instruction'] = apply_dp_to_text(result['evolved_instruction'])
                
            # Add privacy metadata
            result['privacy_processed'] = True
            result['dp_applied'] = True
            
            return result
            
        # Apply the processing to all examples
        return results.map(dp_process)
    
    def _apply_content_filtering(self, results: Results) -> Results:
        """
        Apply content filtering to remove sensitive information
        
        Args:
            results: Results to process
            
        Returns:
            Processed Results
        """
        logger.info("Applying content filtering")
        
        # Create patterns for sensitive terms
        sensitive_patterns = []
        if self.config.sensitive_terms:
            pattern_str = '|'.join(re.escape(term) for term in self.config.sensitive_terms)
            sensitive_patterns.append(re.compile(pattern_str, re.IGNORECASE))
            
        # Add patterns for common PII (simplified)
        # In a real implementation, these would be more sophisticated
        pii_patterns = [
            re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),  # SSN
            re.compile(r'\b\d{16}\b'),  # Credit card (simplified)
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'),  # Phone
        ]
        
        all_patterns = sensitive_patterns + pii_patterns
        
        def filter_sensitive_content(text: str) -> str:
            """Replace sensitive content with [REDACTED]"""
            processed_text = text
            for pattern in all_patterns:
                processed_text = pattern.sub('[REDACTED]', processed_text)
            return processed_text
            
        def content_filter_process(example: Dict[str, Any]) -> Dict[str, Any]:
            result = example.copy()
            
            # Filter sensitive content from relevant text fields
            if 'instruction' in result:
                result['instruction'] = filter_sensitive_content(result['instruction'])
                
            if 'response' in result:
                result['response'] = filter_sensitive_content(result['response'])
                
            if 'evolved_instruction' in result:
                result['evolved_instruction'] = filter_sensitive_content(result['evolved_instruction'])
                
            # Add privacy metadata
            result['privacy_processed'] = True
            result['content_filtered'] = True
            
            return result
            
        # Apply the processing to all examples
        return results.map(content_filter_process) 