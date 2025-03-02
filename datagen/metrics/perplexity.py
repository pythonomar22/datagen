"""
Perplexity Module

This module provides functions to compute perplexity scores for text,
which is a measure of how well a language model can predict the text.
Lower perplexity often correlates with more natural, fluent text.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
import math

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False

from datagen.results import Results

logger = logging.getLogger(__name__)

# Default small model for perplexity calculation
DEFAULT_MODEL = "distilgpt2"
DEFAULT_MAX_LENGTH = 1024  # Maximum sequence length to process


def compute_perplexity(
    dataset: Results,
    model_name: str = DEFAULT_MODEL,
    field: Optional[str] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 8,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute perplexity scores for text in a dataset.
    
    Args:
        dataset: Results object containing the dataset to analyze
        model_name: Name of the pre-trained language model to use for perplexity
        field: Specific field to analyze (e.g., 'instruction', 'response'). If None, all text fields are analyzed.
        max_length: Maximum token length to process
        batch_size: Batch size for processing
        device: Device to use for computation ('cpu', 'cuda', etc.). If None, will use CUDA if available.
        
    Returns:
        Dictionary containing perplexity scores
    """
    if not transformers_available:
        logger.warning("Transformers not available. Cannot compute perplexity.")
        return {"error": "transformers not installed"}
    
    if len(dataset) == 0:
        logger.warning("Empty dataset provided for perplexity calculation. Returning empty scores.")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = dataset.to_dataframe()
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize results dictionary
    perplexity_scores = {}
    
    # Identify text fields if not specified
    text_fields = []
    if field:
        if field in df.columns:
            text_fields.append(field)
        else:
            logger.warning(f"Field '{field}' not found in dataset. Cannot compute perplexity.")
            return {"error": f"Field '{field}' not found"}
    else:
        for col in df.columns:
            if df[col].dtype == 'object' and pd.api.types.is_string_dtype(df[col]):
                text_fields.append(col)
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model '{model_name}' for perplexity calculation")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Calculate perplexity for each field
        for field in text_fields:
            field_perplexity = calculate_field_perplexity(
                df[field].tolist(),
                model,
                tokenizer,
                max_length,
                batch_size,
                device
            )
            perplexity_scores[field] = field_perplexity
        
        # Add metadata
        perplexity_scores["metadata"] = {
            "model_used": model_name,
            "device": device
        }
        
        return perplexity_scores
    
    except Exception as e:
        logger.error(f"Error computing perplexity: {str(e)}")
        return {"error": str(e)}


def calculate_field_perplexity(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    device: str
) -> Dict[str, Any]:
    """
    Calculate perplexity scores for a list of texts.
    
    Args:
        texts: List of text strings to analyze
        model: Pre-trained language model
        tokenizer: Tokenizer for the model
        max_length: Maximum token length to process
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Dictionary containing perplexity scores for the texts
    """
    if not texts:
        return {}
    
    # Initialize lists to store results
    all_perplexities = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        batch_perplexities = []
        
        with torch.no_grad():
            for text in batch_texts:
                # Skip empty texts
                if not text.strip():
                    continue
                
                # Tokenize the text
                encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                encodings = {k: v.to(device) for k, v in encodings.items()}
                
                # Calculate perplexity
                input_ids = encodings["input_ids"]
                target_ids = input_ids.clone()
                
                outputs = model(**encodings)
                logits = outputs.logits
                
                # Shift so that each token is compared to the next
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Calculate perplexity from loss
                perplexity = math.exp(loss.item())
                batch_perplexities.append(perplexity)
        
        all_perplexities.extend(batch_perplexities)
    
    # Calculate statistics if we have any valid perplexities
    if all_perplexities:
        mean_perplexity = float(np.mean(all_perplexities))
        median_perplexity = float(np.median(all_perplexities))
        min_perplexity = float(np.min(all_perplexities))
        max_perplexity = float(np.max(all_perplexities))
        std_dev = float(np.std(all_perplexities))
        
        # Calculate histogram of perplexity values
        hist, bin_edges = np.histogram(all_perplexities, bins=10)
        perplexity_distribution = {
            "bin_edges": bin_edges.tolist(),
            "frequency": hist.tolist()
        }
        
        # Identify outliers (more than 2 standard deviations from mean)
        outlier_threshold = mean_perplexity + 2 * std_dev
        outlier_count = sum(1 for p in all_perplexities if p > outlier_threshold)
        outlier_percentage = float(outlier_count / len(all_perplexities) * 100)
        
        return {
            "mean": mean_perplexity,
            "median": median_perplexity,
            "min": min_perplexity,
            "max": max_perplexity,
            "std_dev": std_dev,
            "distribution": perplexity_distribution,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "sample_count": len(all_perplexities)
        }
    else:
        return {"error": "No valid texts for perplexity calculation"} 