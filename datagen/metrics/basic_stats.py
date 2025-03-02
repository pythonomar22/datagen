"""
Basic Statistics Module

This module provides functions to compute basic statistical metrics for datasets,
such as counts, lengths, distributions, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

from datagen.results import Results

logger = logging.getLogger(__name__)


def compute_basic_stats(dataset: Results) -> Dict[str, Any]:
    """
    Compute basic statistics for a dataset.
    
    Args:
        dataset: Results object containing the dataset to analyze
        
    Returns:
        Dictionary containing basic statistics
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset provided for basic statistics. Returning empty stats.")
        return {"count": 0}
    
    # Convert to DataFrame for easier analysis
    df = dataset.to_dataframe()
    
    # Initialize results dictionary
    stats = {
        "count": len(df),
        "fields": {}
    }
    
    # Calculate statistics for each text field (instruction, response, etc.)
    text_fields = []
    for col in df.columns:
        if df[col].dtype == 'object' and pd.api.types.is_string_dtype(df[col]):
            text_fields.append(col)
    
    # Calculate length statistics for each text field
    for field in text_fields:
        field_stats = calculate_text_field_stats(df, field)
        stats["fields"][field] = field_stats
    
    # Calculate uniqueness statistics
    stats["uniqueness"] = calculate_uniqueness_stats(df, text_fields)
    
    # Calculate source distribution if available
    if "source" in df.columns:
        stats["source_distribution"] = df["source"].value_counts().to_dict()
    
    # Calculate batch distribution if available
    if "batch" in df.columns:
        stats["batch_distribution"] = df["batch"].value_counts().to_dict()
    
    # Overall dataset statistics
    stats["total_tokens_estimate"] = sum(
        stats["fields"][field]["total_tokens_estimate"] 
        for field in text_fields 
        if "total_tokens_estimate" in stats["fields"][field]
    )
    
    return stats


def calculate_text_field_stats(df: pd.DataFrame, field: str) -> Dict[str, Any]:
    """
    Calculate statistics for a specific text field in the dataset.
    
    Args:
        df: DataFrame containing the dataset
        field: Column name of the text field to analyze
        
    Returns:
        Dictionary containing statistics for the field
    """
    # Get lengths of all entries in the field
    lengths = df[field].str.len()
    
    # Calculate basic length statistics
    field_stats = {
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(lengths.median()),
        "std_dev_length": float(lengths.std()),
        "total_characters": int(lengths.sum()),
    }
    
    # Calculate length distribution
    bins = np.linspace(0, lengths.max(), min(20, lengths.max() // 10 + 1))
    hist, bin_edges = np.histogram(lengths, bins=bins)
    field_stats["length_distribution"] = {
        "bin_edges": bin_edges.tolist(),
        "frequency": hist.tolist()
    }
    
    # Calculate word count statistics (rough approximation)
    word_counts = df[field].str.split().str.len()
    field_stats["min_words"] = int(word_counts.min())
    field_stats["max_words"] = int(word_counts.max())
    field_stats["mean_words"] = float(word_counts.mean())
    field_stats["total_words"] = int(word_counts.sum())
    
    # Estimate number of tokens (rough approximation)
    # Assuming average of 4 characters per token, which is typical for English
    field_stats["total_tokens_estimate"] = int(field_stats["total_characters"] / 4)
    
    # Check for empty entries
    empty_count = (df[field].str.len() == 0).sum()
    field_stats["empty_count"] = int(empty_count)
    field_stats["empty_percentage"] = float(empty_count / len(df) * 100)
    
    return field_stats


def calculate_uniqueness_stats(df: pd.DataFrame, text_fields: List[str]) -> Dict[str, Any]:
    """
    Calculate uniqueness statistics for text fields.
    
    Args:
        df: DataFrame containing the dataset
        text_fields: List of column names for text fields to analyze
        
    Returns:
        Dictionary containing uniqueness statistics
    """
    uniqueness_stats = {}
    
    for field in text_fields:
        # Count unique entries
        total_count = len(df)
        unique_count = df[field].nunique()
        
        uniqueness_stats[field] = {
            "unique_count": int(unique_count),
            "unique_percentage": float(unique_count / total_count * 100),
            "duplication_rate": float((total_count - unique_count) / total_count * 100)
        }
        
        # Find most common entries (if any duplicates exist)
        if unique_count < total_count:
            most_common = df[field].value_counts().head(5).to_dict()
            uniqueness_stats[field]["most_common"] = {
                str(k): int(v) for k, v in most_common.items()
            }
    
    # If both instruction and response fields exist, calculate pair uniqueness
    if "instruction" in text_fields and "response" in text_fields:
        pair_uniqueness = df.duplicated(subset=["instruction", "response"]).sum()
        uniqueness_stats["instruction_response_pair"] = {
            "unique_count": int(len(df) - pair_uniqueness),
            "unique_percentage": float((len(df) - pair_uniqueness) / len(df) * 100),
            "duplication_rate": float(pair_uniqueness / len(df) * 100)
        }
    
    return uniqueness_stats 