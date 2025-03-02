"""
Diversity Metrics Module

This module provides functions to compute diversity metrics for datasets,
such as n-gram uniqueness, semantic diversity via embeddings, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
import re
import logging
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.util import ngrams
    from nltk.tokenize import word_tokenize
    nltk_available = True
except ImportError:
    nltk_available = False

try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    embedding_available = True
except ImportError:
    embedding_available = False

from datagen.results import Results

logger = logging.getLogger(__name__)


def compute_diversity_metrics(dataset: Results) -> Dict[str, Any]:
    """
    Compute diversity metrics for a dataset.
    
    Args:
        dataset: Results object containing the dataset to analyze
        
    Returns:
        Dictionary containing diversity metrics
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset provided for diversity metrics. Returning empty metrics.")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = dataset.to_dataframe()
    
    # Initialize results dictionary
    metrics = {}
    
    # Identify text fields
    text_fields = []
    for col in df.columns:
        if df[col].dtype == 'object' and pd.api.types.is_string_dtype(df[col]):
            text_fields.append(col)
    
    # Calculate n-gram uniqueness for each text field
    metrics["ngram_uniqueness"] = {}
    for field in text_fields:
        metrics["ngram_uniqueness"][field] = calculate_ngram_uniqueness(df[field].tolist())
    
    # Calculate semantic diversity if sentence-transformers is available
    if embedding_available:
        try:
            metrics["semantic_diversity"] = {}
            for field in text_fields:
                metrics["semantic_diversity"][field] = calculate_semantic_diversity(df[field].tolist())
        except Exception as e:
            logger.warning(f"Failed to calculate semantic diversity: {str(e)}")
            metrics["semantic_diversity"] = {"error": str(e)}
    else:
        logger.info("Sentence Transformers not available. Skipping semantic diversity calculation.")
        metrics["semantic_diversity"] = {"error": "sentence-transformers not installed"}
    
    # Calculate string similarity-based diversity
    metrics["similarity_metrics"] = {}
    for field in text_fields:
        metrics["similarity_metrics"][field] = calculate_similarity_metrics(df[field].tolist())
    
    # Calculate vocabulary richness
    metrics["vocabulary_richness"] = {}
    for field in text_fields:
        metrics["vocabulary_richness"][field] = calculate_vocabulary_richness(df[field].tolist())
    
    return metrics


def calculate_ngram_uniqueness(texts: List[str], max_n: int = 4) -> Dict[str, Any]:
    """
    Calculate n-gram uniqueness for a list of texts.
    
    Args:
        texts: List of text strings to analyze
        max_n: Maximum n-gram size to consider
        
    Returns:
        Dictionary containing n-gram uniqueness metrics
    """
    if not texts:
        return {}
    
    results = {}
    
    # If NLTK is available, use it for more accurate tokenization
    if nltk_available:
        try:
            # Download tokenizers if not already downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Tokenize texts
            tokenized_texts = [word_tokenize(text.lower()) for text in texts]
            
            # Calculate n-gram uniqueness for different n values
            for n in range(1, min(max_n + 1, 6)):  # Limit to reasonable n values
                all_ngrams = []
                for tokens in tokenized_texts:
                    if len(tokens) >= n:
                        all_ngrams.extend(list(ngrams(tokens, n)))
                
                if all_ngrams:
                    unique_ngrams = set(all_ngrams)
                    total_ngrams = len(all_ngrams)
                    unique_ratio = len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0
                    
                    results[f"{n}-gram"] = {
                        "unique_count": len(unique_ngrams),
                        "total_count": total_ngrams,
                        "unique_ratio": float(unique_ratio),
                        "diversity_score": float(1 - (1 / unique_ratio if unique_ratio > 0 else 0))
                    }
        except Exception as e:
            logger.warning(f"Failed to calculate n-gram uniqueness using NLTK: {str(e)}")
    
    # Fallback to sklearn's CountVectorizer if NLTK fails or is not available
    if not results:
        for n in range(1, min(max_n + 1, 6)):
            try:
                # Create a vectorizer to extract n-grams
                vectorizer = CountVectorizer(ngram_range=(n, n), lowercase=True)
                X = vectorizer.fit_transform(texts)
                
                # Get n-gram counts
                ngram_counts = np.asarray(X.sum(axis=0)).flatten()
                
                # Calculate uniqueness metrics
                unique_ngrams = (ngram_counts == 1).sum()
                total_unique_ngrams = len(ngram_counts)
                total_ngrams = ngram_counts.sum()
                
                unique_ratio = total_unique_ngrams / total_ngrams if total_ngrams > 0 else 0
                
                results[f"{n}-gram"] = {
                    "unique_count": int(total_unique_ngrams),
                    "total_count": int(total_ngrams),
                    "unique_ratio": float(unique_ratio),
                    "diversity_score": float(1 - (1 / unique_ratio if unique_ratio > 0 else 0))
                }
            except Exception as e:
                logger.warning(f"Failed to calculate {n}-gram uniqueness: {str(e)}")
    
    return results


def calculate_semantic_diversity(texts: List[str], 
                                 model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Calculate semantic diversity of texts using embeddings.
    
    Args:
        texts: List of text strings to analyze
        model_name: Name of the sentence-transformer model to use
        
    Returns:
        Dictionary containing semantic diversity metrics
    """
    if not embedding_available:
        return {"error": "sentence-transformers not installed"}
    
    if not texts:
        return {}
    
    # If texts are too long, truncate to avoid excessive processing time
    max_seq_length = 512
    truncated_texts = [text[:max_seq_length * 4] for text in texts]  # 4 chars per token (approx)
    
    try:
        # Load model
        model = SentenceTransformer(model_name)
        
        # Get embeddings
        embeddings = model.encode(truncated_texts, show_progress_bar=False)
        
        # Calculate pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        # Calculate diversity metrics from the similarity matrix
        np.fill_diagonal(similarities, 0)  # Ignore self-similarity
        
        mean_similarity = float(similarities.mean())
        max_similarity = float(similarities.max())
        min_similarity = float(similarities[similarities > 0].min()) if similarities.any() > 0 else 0
        
        # Calculate various diversity scores
        diversity_score = float(1 - mean_similarity)
        
        # Calculate similarity distribution
        hist, bin_edges = np.histogram(similarities.flatten(), bins=10, range=(0, 1))
        similarity_distribution = {
            "bin_edges": bin_edges.tolist(),
            "frequency": hist.tolist()
        }
        
        return {
            "mean_similarity": mean_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,
            "diversity_score": diversity_score,
            "similarity_distribution": similarity_distribution,
            "model_used": model_name
        }
    except Exception as e:
        logger.warning(f"Failed to calculate semantic diversity: {str(e)}")
        return {"error": str(e)}


def calculate_similarity_metrics(texts: List[str], 
                                 sample_size: int = 100) -> Dict[str, Any]:
    """
    Calculate similarity-based diversity metrics using string methods.
    
    Args:
        texts: List of text strings to analyze
        sample_size: Maximum number of texts to sample for pairwise comparisons
        
    Returns:
        Dictionary containing similarity-based diversity metrics
    """
    if not texts:
        return {}
    
    # If there are too many texts, sample a subset to avoid excessive computation
    if len(texts) > sample_size:
        import random
        random.seed(42)  # For reproducibility
        sampled_texts = random.sample(texts, sample_size)
    else:
        sampled_texts = texts
    
    # Jaccard similarity for character 3-grams
    try:
        jaccard_similarities = []
        char_3grams = []
        
        # Extract character 3-grams for each text
        for text in sampled_texts:
            text = text.lower()
            if len(text) < 3:
                char_3grams.append(set())
                continue
            
            grams = set()
            for i in range(len(text) - 2):
                grams.add(text[i:i+3])
            char_3grams.append(grams)
        
        # Calculate pairwise Jaccard similarities
        for i in range(len(char_3grams)):
            for j in range(i+1, len(char_3grams)):
                if not char_3grams[i] or not char_3grams[j]:
                    continue
                
                intersection = len(char_3grams[i] & char_3grams[j])
                union = len(char_3grams[i] | char_3grams[j])
                
                if union > 0:
                    jaccard_similarities.append(intersection / union)
        
        if jaccard_similarities:
            mean_jaccard = float(np.mean(jaccard_similarities))
            jaccard_diversity = float(1 - mean_jaccard)
        else:
            mean_jaccard = 0
            jaccard_diversity = 1
        
        return {
            "char_3gram_jaccard": {
                "mean_similarity": mean_jaccard,
                "diversity_score": jaccard_diversity
            }
        }
    except Exception as e:
        logger.warning(f"Failed to calculate string similarity metrics: {str(e)}")
        return {"error": str(e)}


def calculate_vocabulary_richness(texts: List[str]) -> Dict[str, Any]:
    """
    Calculate vocabulary richness metrics.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        Dictionary containing vocabulary richness metrics
    """
    if not texts:
        return {}
    
    try:
        # Concatenate all texts
        combined_text = " ".join(texts).lower()
        
        # Tokenize into words (simple whitespace tokenization)
        words = re.findall(r'\b\w+\b', combined_text)
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Calculate basic vocabulary metrics
        total_words = len(words)
        unique_words = len(word_counts)
        
        # Calculate type-token ratio (TTR)
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Calculate logarithmic TTR
        log_ttr = unique_words / math.log(total_words) if total_words > 1 else 0
        
        # Most common words
        most_common = word_counts.most_common(20)
        
        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": float(ttr),
            "log_ttr": float(log_ttr),
            "most_common_words": {word: count for word, count in most_common}
        }
    except Exception as e:
        logger.warning(f"Failed to calculate vocabulary richness: {str(e)}")
        return {"error": str(e)} 