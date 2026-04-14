"""
Text preprocessing module for emotion classification system.

This module provides the TextPreprocessor class that handles all text normalization,
tokenization, and feature extraction needed for the Naive Bayes emotion classifier.
"""

import re
import string
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class TextPreprocessor:
    """
    Handles text preprocessing for emotion classification.
    
    This class provides consistent text preprocessing between training and prediction
    phases, including normalization, tokenization, stop word removal, and feature
    extraction suitable for Naive Bayes classification.
    """
    
    def __init__(self, remove_stopwords: bool = True, min_word_freq: int = 1, 
                 max_features: int = 10000):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove common English stop words
            min_word_freq: Minimum frequency for words to be included in vocabulary
            max_features: Maximum number of features in vocabulary
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_freq = min_word_freq
        self.max_features = max_features
        
        # Initialize stop words set
        self.stop_words = self._get_stop_words() if remove_stopwords else set()
        
        # Vocabulary and feature extraction
        self.vocabulary: Dict[str, int] = {}
        self.word_counts: Counter = Counter()
        self.is_fitted = False
        
        # Use sklearn's CountVectorizer for efficient feature extraction
        self.vectorizer: Optional[CountVectorizer] = None
    
    def _get_stop_words(self) -> Set[str]:
        """
        Get a set of common English stop words.
        
        Returns:
            Set of stop words to be removed during preprocessing
        """
        # Common English stop words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'through', 'during', 'before', 'after', 'above', 'below', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now'
        }
        return stop_words
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to lowercase and removing punctuation.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep only letters, numbers and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize normalized text into individual words.
        
        Args:
            text: Normalized text string
            
        Returns:
            List of tokens (words)
        """
        # Split on whitespace
        tokens = text.split()
        
        # Remove stop words if configured
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filter out very short tokens (single characters)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for feature extraction
        """
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Tokenize and rejoin
        tokens = self.tokenize(normalized)
        
        return ' '.join(tokens)
    
    def fit_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training text samples
        """
        # Preprocess all texts
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Initialize CountVectorizer with our parameters
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_word_freq,
            token_pattern=r'\b\w+\b',  # Word boundaries
            lowercase=False,  # Already lowercased in preprocessing
            stop_words=None   # Already handled in preprocessing
        )
        
        # Fit the vectorizer
        self.vectorizer.fit(processed_texts)
        
        # Store vocabulary for reference
        self.vocabulary = self.vectorizer.vocabulary_
        self.is_fitted = True
    
    def extract_features(self, texts: List[str]) -> csr_matrix:
        """
        Convert texts to feature vectors for Naive Bayes classification.
        
        Args:
            texts: List of text samples to convert
            
        Returns:
            Sparse matrix where each row is a feature vector for one text
            
        Raises:
            ValueError: If vocabulary hasn't been fitted yet
        """
        if not self.is_fitted or self.vectorizer is None:
            raise ValueError("Vocabulary must be fitted before extracting features. "
                           "Call fit_vocabulary() first.")
        
        # Preprocess all texts
        processed_texts = [self.preprocess(text) for text in texts]
        
        # Transform to feature vectors
        feature_matrix = self.vectorizer.transform(processed_texts)
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names (vocabulary words).
        
        Returns:
            List of words in the vocabulary
            
        Raises:
            ValueError: If vocabulary hasn't been fitted yet
        """
        if not self.is_fitted or self.vectorizer is None:
            raise ValueError("Vocabulary must be fitted before getting feature names.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the fitted vocabulary.
        
        Returns:
            Number of words in vocabulary
        """
        if not self.is_fitted:
            return 0
        return len(self.vocabulary)
    
    def transform_single(self, text: str) -> csr_matrix:
        """
        Transform a single text into feature vector.
        
        Args:
            text: Single text sample
            
        Returns:
            Sparse matrix with one row (feature vector)
        """
        return self.extract_features([text])
    
    def get_preprocessing_info(self) -> Dict[str, any]:
        """
        Get information about preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing settings and statistics
        """
        return {
            'remove_stopwords': self.remove_stopwords,
            'min_word_freq': self.min_word_freq,
            'max_features': self.max_features,
            'vocabulary_size': self.get_vocabulary_size(),
            'is_fitted': self.is_fitted,
            'stop_words_count': len(self.stop_words)
        }