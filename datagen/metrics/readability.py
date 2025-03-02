"""
Readability Module

This module provides functions to compute readability metrics for text,
such as Flesch-Kincaid Grade Level, SMOG Index, and others.
These metrics assess how easily a human can understand the text.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, Any, List, Optional, Tuple
import logging
import math

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    nltk_available = True
except ImportError:
    nltk_available = False

from datagen.results import Results

logger = logging.getLogger(__name__)


def compute_readability_scores(dataset: Results, field: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute readability scores for text in a dataset.
    
    Args:
        dataset: Results object containing the dataset to analyze
        field: Specific field to analyze (e.g., 'instruction', 'response'). If None, all text fields are analyzed.
        
    Returns:
        Dictionary containing readability scores
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset provided for readability calculation. Returning empty scores.")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = dataset.to_dataframe()
    
    # Initialize results dictionary
    readability_scores = {}
    
    # Identify text fields if not specified
    text_fields = []
    if field:
        if field in df.columns:
            text_fields.append(field)
        else:
            logger.warning(f"Field '{field}' not found in dataset. Cannot compute readability.")
            return {"error": f"Field '{field}' not found"}
    else:
        for col in df.columns:
            if df[col].dtype == 'object' and pd.api.types.is_string_dtype(df[col]):
                text_fields.append(col)
    
    # Calculate readability for each field
    for field in text_fields:
        field_scores = calculate_field_readability(df[field].tolist())
        readability_scores[field] = field_scores
    
    # Calculate aggregate scores across all fields if there's more than one
    if len(text_fields) > 1:
        combined_texts = []
        for field in text_fields:
            combined_texts.extend(df[field].tolist())
        
        readability_scores["combined"] = calculate_field_readability(combined_texts)
    
    return readability_scores


def calculate_field_readability(texts: List[str]) -> Dict[str, Any]:
    """
    Calculate readability scores for a list of texts.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        Dictionary containing readability scores for the texts
    """
    if not texts:
        return {}
    
    # Prepare NLTK resources if available
    if nltk_available:
        try:
            # Download tokenizers if not already downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK resources: {str(e)}")
    
    # Calculate readability scores for each text
    flesch_reading_ease_scores = []
    flesch_kincaid_grade_scores = []
    smog_index_scores = []
    coleman_liau_index_scores = []
    automated_readability_index_scores = []
    avg_sentence_length_scores = []
    avg_word_length_scores = []
    
    for text in texts:
        # Skip empty or very short texts
        if not text or len(text) < 10:
            continue
        
        # Calculate sentence and word counts
        sentence_count, word_count, syllable_count, char_count, complex_word_count = count_text_elements(text)
        
        # Skip texts with insufficient words or sentences
        if sentence_count < 1 or word_count < 3:
            continue
        
        # Calculate average sentence and word lengths
        avg_sentence_length = word_count / sentence_count
        avg_word_length = char_count / word_count
        
        # Calculate Flesch Reading Ease
        # Range: 0-100, higher score = easier to read
        # 90-100: Very Easy, 80-89: Easy, 70-79: Fairly Easy, etc.
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (syllable_count / word_count))
        flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp to 0-100 range
        
        # Calculate Flesch-Kincaid Grade Level
        # Corresponds to grade level: 8.0 = 8th grade
        flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * (syllable_count / word_count)) - 15.59
        flesch_kincaid_grade = max(0, flesch_kincaid_grade)  # Ensure non-negative
        
        # Calculate SMOG Index
        # Simple Measure of Gobbledygook estimates years of education needed to understand text
        if sentence_count >= 30:
            smog_index = 1.043 * math.sqrt(complex_word_count * (30 / sentence_count)) + 3.1291
        else:
            # Adjustment for short texts
            smog_index = 1.043 * math.sqrt(complex_word_count * (30 / sentence_count)) + 3.1291
        smog_index = max(0, smog_index)  # Ensure non-negative
        
        # Calculate Coleman-Liau Index
        # Grade level based on characters per word and sentences per word
        l = (char_count / word_count) * 100  # Letters per 100 words
        s = (sentence_count / word_count) * 100  # Sentences per 100 words
        coleman_liau_index = 0.0588 * l - 0.296 * s - 15.8
        coleman_liau_index = max(0, coleman_liau_index)  # Ensure non-negative
        
        # Calculate Automated Readability Index (ARI)
        # Grade level based on characters per word and words per sentence
        automated_readability_index = 4.71 * (char_count / word_count) + 0.5 * avg_sentence_length - 21.43
        automated_readability_index = max(0, automated_readability_index)  # Ensure non-negative
        
        # Append scores to lists
        flesch_reading_ease_scores.append(flesch_reading_ease)
        flesch_kincaid_grade_scores.append(flesch_kincaid_grade)
        smog_index_scores.append(smog_index)
        coleman_liau_index_scores.append(coleman_liau_index)
        automated_readability_index_scores.append(automated_readability_index)
        avg_sentence_length_scores.append(avg_sentence_length)
        avg_word_length_scores.append(avg_word_length)
    
    # Calculate aggregated metrics if we have any valid scores
    if flesch_reading_ease_scores:
        aggregated_scores = {
            "flesch_reading_ease": {
                "mean": float(np.mean(flesch_reading_ease_scores)),
                "median": float(np.median(flesch_reading_ease_scores)),
                "min": float(np.min(flesch_reading_ease_scores)),
                "max": float(np.max(flesch_reading_ease_scores)),
                "std_dev": float(np.std(flesch_reading_ease_scores)),
                "interpretation": interpret_flesch_reading_ease(np.mean(flesch_reading_ease_scores))
            },
            "flesch_kincaid_grade": {
                "mean": float(np.mean(flesch_kincaid_grade_scores)),
                "median": float(np.median(flesch_kincaid_grade_scores)),
                "min": float(np.min(flesch_kincaid_grade_scores)),
                "max": float(np.max(flesch_kincaid_grade_scores)),
                "std_dev": float(np.std(flesch_kincaid_grade_scores))
            },
            "smog_index": {
                "mean": float(np.mean(smog_index_scores)),
                "median": float(np.median(smog_index_scores)),
                "min": float(np.min(smog_index_scores)),
                "max": float(np.max(smog_index_scores)),
                "std_dev": float(np.std(smog_index_scores))
            },
            "coleman_liau_index": {
                "mean": float(np.mean(coleman_liau_index_scores)),
                "median": float(np.median(coleman_liau_index_scores)),
                "min": float(np.min(coleman_liau_index_scores)),
                "max": float(np.max(coleman_liau_index_scores)),
                "std_dev": float(np.std(coleman_liau_index_scores))
            },
            "automated_readability_index": {
                "mean": float(np.mean(automated_readability_index_scores)),
                "median": float(np.median(automated_readability_index_scores)),
                "min": float(np.min(automated_readability_index_scores)),
                "max": float(np.max(automated_readability_index_scores)),
                "std_dev": float(np.std(automated_readability_index_scores))
            },
            "avg_sentence_length": {
                "mean": float(np.mean(avg_sentence_length_scores)),
                "median": float(np.median(avg_sentence_length_scores)),
                "min": float(np.min(avg_sentence_length_scores)),
                "max": float(np.max(avg_sentence_length_scores)),
                "std_dev": float(np.std(avg_sentence_length_scores))
            },
            "avg_word_length": {
                "mean": float(np.mean(avg_word_length_scores)),
                "median": float(np.median(avg_word_length_scores)),
                "min": float(np.min(avg_word_length_scores)),
                "max": float(np.max(avg_word_length_scores)),
                "std_dev": float(np.std(avg_word_length_scores))
            },
            "sample_count": len(flesch_reading_ease_scores)
        }
        
        # Calculate average grade level across multiple metrics
        grade_level_metrics = ["flesch_kincaid_grade", "smog_index", "coleman_liau_index", "automated_readability_index"]
        grade_means = [aggregated_scores[metric]["mean"] for metric in grade_level_metrics]
        aggregated_scores["consensus_grade_level"] = float(np.mean(grade_means))
        
        return aggregated_scores
    else:
        return {"error": "No valid texts for readability calculation"}


def count_text_elements(text: str) -> Tuple[int, int, int, int, int]:
    """
    Count sentences, words, syllables, characters, and complex words in a text.
    
    Args:
        text: Text string to analyze
        
    Returns:
        Tuple containing counts of (sentences, words, syllables, characters, complex words)
    """
    # Clean the text
    text = text.strip()
    
    # Count sentences
    if nltk_available:
        try:
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
        except Exception:
            # Fallback to simple regex if NLTK fails
            sentence_count = len(re.split(r'[.!?]+', text))
    else:
        # Simple sentence counting using regex
        sentence_count = len(re.split(r'[.!?]+', text))
    
    # Ensure at least 1 sentence
    sentence_count = max(1, sentence_count)
    
    # Count characters (excluding spaces)
    char_count = len(re.sub(r'\s', '', text))
    
    # Count words
    if nltk_available:
        try:
            words = word_tokenize(text)
            # Filter out punctuation
            words = [word for word in words if re.match(r'\w', word)]
            word_count = len(words)
        except Exception:
            # Fallback to simple regex if NLTK fails
            word_count = len(re.findall(r'\b\w+\b', text))
    else:
        # Simple word counting using regex
        word_count = len(re.findall(r'\b\w+\b', text))
    
    # Ensure at least 1 word
    word_count = max(1, word_count)
    
    # Count syllables and complex words
    syllable_count = 0
    complex_word_count = 0
    
    for word in re.findall(r'\b\w+\b', text.lower()):
        syllables = count_syllables(word)
        syllable_count += syllables
        
        # Complex words have 3 or more syllables (excluding -es, -ed, -ing suffixes)
        if syllables >= 3 and not word.endswith(('es', 'ed', 'ing')):
            complex_word_count += 1
    
    return sentence_count, word_count, syllable_count, char_count, complex_word_count


def count_syllables(word: str) -> int:
    """
    Count the number of syllables in a word.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables
    """
    # Clean the word
    word = word.lower().strip()
    
    # If the word is empty, return 0
    if not word:
        return 0
    
    # Count vowel groups
    count = 0
    vowels = "aeiouy"
    
    # If word starts with 'y', don't count the first 'y' as a vowel
    if word[0] in vowels:
        count += 1
    
    # Count vowel groups (sequences of vowels)
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    
    # Adjust for certain patterns
    
    # If the word ends with 'e', don't count it as a syllable unless it's 'le'
    if word.endswith('e') and len(word) > 2 and word[-2] not in vowels:
        if not (word.endswith('le') and word[-3] not in vowels):
            count -= 1
    
    # If word ends with 'ed' and the 'e' is silent, don't count it
    if word.endswith('ed') and len(word) > 2 and word[-3] not in vowels:
        count -= 1
    
    # Count some common suffixes like '-ion' as a syllable
    if word.endswith(('ion', 'ious', 'ia', 'ier')):
        count += 1
    
    # Ensure at least 1 syllable per word
    return max(1, count)


def interpret_flesch_reading_ease(score: float) -> str:
    """
    Interpret a Flesch Reading Ease score.
    
    Args:
        score: Flesch Reading Ease score (0-100)
        
    Returns:
        String interpretation of the score
    """
    if score >= 90:
        return "Very Easy - 5th grade level"
    elif score >= 80:
        return "Easy - 6th grade level"
    elif score >= 70:
        return "Fairly Easy - 7th grade level"
    elif score >= 60:
        return "Standard - 8th to 9th grade level"
    elif score >= 50:
        return "Fairly Difficult - 10th to 12th grade level"
    elif score >= 30:
        return "Difficult - College level"
    else:
        return "Very Difficult - College graduate level" 