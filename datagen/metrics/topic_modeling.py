"""
Topic Modeling Module

This module provides functions to extract topics from a dataset using
unsupervised topic modeling techniques (e.g., LDA or BERTopic).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import re
from collections import Counter

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    gensim_available = True
except ImportError:
    gensim_available = False

try:
    from bertopic import BERTopic
    bertopic_available = True
except ImportError:
    bertopic_available = False

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk_available = True
except ImportError:
    nltk_available = False

from datagen.results import Results

logger = logging.getLogger(__name__)


def extract_topics(
    dataset: Results,
    method: str = "auto",
    field: Optional[str] = None,
    num_topics: int = 10,
    top_n_words: int = 10
) -> Dict[str, Any]:
    """
    Extract topics from a dataset using topic modeling.
    
    Args:
        dataset: Results object containing the dataset to analyze
        method: Topic modeling method to use ('lda', 'bertopic', or 'auto')
        field: Specific field to analyze (e.g., 'instruction', 'response'). If None, all text fields are analyzed.
        num_topics: Number of topics to extract
        top_n_words: Number of top words to include for each topic
        
    Returns:
        Dictionary containing extracted topics and their weights
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset provided for topic extraction. Returning empty topics.")
        return {}
    
    # Convert to DataFrame for easier analysis
    df = dataset.to_dataframe()
    
    # Determine the best available method if 'auto' is specified
    if method == "auto":
        if bertopic_available:
            method = "bertopic"
        elif gensim_available:
            method = "lda"
        else:
            method = "frequency"
            logger.warning("No topic modeling libraries available. Falling back to simple word frequency analysis.")
    
    # Identify text fields if not specified
    text_fields = []
    if field:
        if field in df.columns:
            text_fields.append(field)
        else:
            logger.warning(f"Field '{field}' not found in dataset. Cannot extract topics.")
            return {"error": f"Field '{field}' not found"}
    else:
        for col in df.columns:
            if df[col].dtype == 'object' and pd.api.types.is_string_dtype(df[col]):
                text_fields.append(col)
    
    # Extract topics for each field
    topics_by_field = {}
    
    for field in text_fields:
        texts = df[field].tolist()
        
        if method == "bertopic" and bertopic_available:
            topics = extract_topics_bertopic(texts, num_topics, top_n_words)
        elif method == "lda" and gensim_available:
            topics = extract_topics_lda(texts, num_topics, top_n_words)
        else:
            topics = extract_topics_frequency(texts, num_topics, top_n_words)
        
        topics_by_field[field] = topics
    
    # Combine topics from all fields if there's more than one
    if len(text_fields) > 1:
        combined_texts = []
        for field in text_fields:
            combined_texts.extend(df[field].tolist())
        
        if method == "bertopic" and bertopic_available:
            topics_by_field["combined"] = extract_topics_bertopic(combined_texts, num_topics, top_n_words)
        elif method == "lda" and gensim_available:
            topics_by_field["combined"] = extract_topics_lda(combined_texts, num_topics, top_n_words)
        else:
            topics_by_field["combined"] = extract_topics_frequency(combined_texts, num_topics, top_n_words)
    
    # Add metadata
    topics_by_field["metadata"] = {
        "method_used": method,
        "num_topics": num_topics,
        "top_n_words": top_n_words
    }
    
    return topics_by_field


def extract_topics_bertopic(
    texts: List[str],
    num_topics: int = 10,
    top_n_words: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract topics using BERTopic.
    
    Args:
        texts: List of text strings to analyze
        num_topics: Number of topics to extract
        top_n_words: Number of top words to include for each topic
        
    Returns:
        List of (topic_label, weight) tuples
    """
    if not bertopic_available:
        logger.warning("BERTopic not available. Using fallback method.")
        return extract_topics_frequency(texts, num_topics, top_n_words)
    
    try:
        # Create BERTopic model
        topic_model = BERTopic(nr_topics=num_topics, language="english")
        
        # Fit the model on the texts
        topics, probs = topic_model.fit_transform(texts)
        
        # Get topic info and format the results
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]  # Filter out the -1 topic (outliers)
        
        # Limit to the requested number of topics
        topic_info = topic_info.head(num_topics)
        
        # Extract topic labels and weights
        results = []
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue
                
            # Get topic words and weights
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Create a label from the top words
                label = " + ".join([word for word, _ in topic_words[:min(3, len(topic_words))]])
                
                # Calculate topic weight based on count and representativeness
                topic_weight = float(row['Count'] / len(texts))
                
                results.append((label, topic_weight))
        
        # Sort by weight and limit to the requested number
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_topics]
    
    except Exception as e:
        logger.warning(f"Error in BERTopic topic extraction: {str(e)}")
        return extract_topics_frequency(texts, num_topics, top_n_words)


def extract_topics_lda(
    texts: List[str],
    num_topics: int = 10,
    top_n_words: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract topics using LDA (Latent Dirichlet Allocation).
    
    Args:
        texts: List of text strings to analyze
        num_topics: Number of topics to extract
        top_n_words: Number of top words to include for each topic
        
    Returns:
        List of (topic_label, weight) tuples
    """
    if not gensim_available:
        logger.warning("Gensim not available. Using fallback method.")
        return extract_topics_frequency(texts, num_topics, top_n_words)
    
    try:
        # Prepare NLTK resources if available
        lemmatizer = None
        if nltk_available:
            try:
                # Download necessary resources if not already downloaded
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet', quiet=True)
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Failed to download NLTK resources: {str(e)}")
        
        # Process texts: tokenize, remove stopwords, lemmatize
        processed_texts = []
        for text in texts:
            # Skip empty texts
            if not text.strip():
                continue
                
            # Tokenize, lowercase, and remove short words and stopwords
            tokens = simple_preprocess(text, deacc=True, min_len=3)
            tokens = [token for token in tokens if token not in STOPWORDS]
            
            # Lemmatize if possible
            if lemmatizer:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                
            processed_texts.append(tokens)
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics and format results
        results = []
        topic_weights = []
        
        # Get topic distributions for each document
        doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
        
        # Calculate average weight for each topic
        for i in range(num_topics):
            weights = [weight for doc in doc_topics for topic_id, weight in doc if topic_id == i]
            avg_weight = np.mean(weights) if weights else 0
            topic_weights.append((i, avg_weight))
        
        # Sort topics by weight
        topic_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Create topic labels from top words and get weights
        for topic_id, weight in topic_weights[:num_topics]:
            # Get the top words for this topic
            topic_words = lda_model.show_topic(topic_id, topn=top_n_words)
            
            # Create a label from the top 3 words
            label = " + ".join([word for word, _ in topic_words[:min(3, len(topic_words))]])
            results.append((label, float(weight)))
        
        return results
    
    except Exception as e:
        logger.warning(f"Error in LDA topic extraction: {str(e)}")
        return extract_topics_frequency(texts, num_topics, top_n_words)


def extract_topics_frequency(
    texts: List[str],
    num_topics: int = 10,
    top_n_words: int = 10
) -> List[Tuple[str, float]]:
    """
    Extract topics using simple word frequency analysis.
    
    Args:
        texts: List of text strings to analyze
        num_topics: Number of top word combinations to consider as topics
        top_n_words: Number of top words to include for each topic
        
    Returns:
        List of (topic_label, weight) tuples
    """
    try:
        # Combine all texts
        combined_text = " ".join(texts)
        
        # Tokenize and clean
        # Remove non-alphanumeric, convert to lowercase, and split into words
        words = re.findall(r'\b[a-z]{3,}\b', combined_text.lower())
        
        # List of common stop words to ignore
        stop_words = {
            "the", "and", "of", "to", "a", "in", "for", "is", "on", "that", "by", "this", "with", "you",
            "it", "be", "are", "as", "not", "have", "was", "but", "they", "can", "at", "has", "i", "an",
            "from", "or", "if", "its", "which", "your", "all", "been", "when", "we", "there", "will",
            "would", "their", "what", "so", "some", "more", "these", "than", "such", "also"
        }
        
        # Filter out stop words
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Get the most common words
        top_words = word_counts.most_common(top_n_words)
        
        # Create 'topics' by finding common co-occurrences of words
        # For simplicity, we'll use the top words individually as topics
        total_words = len(words)
        topics = []
        
        for word, count in top_words[:num_topics]:
            weight = count / total_words
            topics.append((word, float(weight)))
        
        return topics
    
    except Exception as e:
        logger.warning(f"Error in frequency-based topic extraction: {str(e)}")
        return [] 