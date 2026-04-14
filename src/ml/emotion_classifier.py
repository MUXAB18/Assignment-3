"""
Emotion classification module using Naive Bayes algorithm.

This module provides the EmotionClassifier class that implements a Naive Bayes
classifier for predicting emotions from text input. It includes training,
prediction, and model serialization capabilities.
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict, Counter
import logging

from .text_processor import TextPreprocessor


class EmotionClassifier:
    """
    Naive Bayes classifier for emotion prediction from text.
    
    This class implements a multinomial Naive Bayes algorithm specifically
    designed for emotion classification. It includes Laplace smoothing,
    model serialization, and comprehensive prediction capabilities.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Emotion Classifier.
        
        Args:
            alpha: Laplace smoothing parameter (default: 1.0)
        """
        self.alpha = alpha
        self.preprocessor = TextPreprocessor()
        
        # Model parameters (set during training)
        self.class_priors: Dict[str, float] = {}
        self.feature_likelihoods: Dict[str, np.ndarray] = {}
        self.classes: List[str] = []
        self.vocabulary_size: int = 0
        self.is_trained: bool = False
        
        # Training statistics
        self.training_stats: Dict[str, Any] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the Naive Bayes model on provided data.
        
        Args:
            texts: List of text samples for training
            labels: List of corresponding emotion labels
            
        Raises:
            ValueError: If texts and labels have different lengths or are empty
        """
        if len(texts) != len(labels):
            raise ValueError(f"Texts and labels must have same length. "
                           f"Got {len(texts)} texts and {len(labels)} labels.")
        
        if len(texts) == 0:
            raise ValueError("Training data cannot be empty.")
        
        self.logger.info(f"Starting training with {len(texts)} samples")
        
        # Get unique classes and sort for consistency
        self.classes = sorted(list(set(labels)))
        self.logger.info(f"Found {len(self.classes)} emotion classes: {self.classes}")
        
        # Fit vocabulary on training texts
        self.preprocessor.fit_vocabulary(texts)
        self.vocabulary_size = self.preprocessor.get_vocabulary_size()
        self.logger.info(f"Vocabulary size: {self.vocabulary_size}")
        
        # Extract features from training texts
        feature_matrix = self.preprocessor.extract_features(texts)
        
        # Calculate class priors P(class)
        self._calculate_class_priors(labels)
        
        # Calculate feature likelihoods P(feature|class)
        self._calculate_feature_likelihoods(feature_matrix, labels)
        
        # Store training statistics
        self._store_training_stats(texts, labels)
        
        self.is_trained = True
        self.logger.info("Training completed successfully")
    
    def _calculate_class_priors(self, labels: List[str]) -> None:
        """Calculate prior probabilities for each class."""
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        self.class_priors = {
            class_name: count / total_samples 
            for class_name, count in label_counts.items()
        }
        
        self.logger.debug(f"Class priors: {self.class_priors}")
    
    def _calculate_feature_likelihoods(self, feature_matrix: csr_matrix, 
                                     labels: List[str]) -> None:
        """Calculate feature likelihoods for each class using Laplace smoothing."""
        # Group samples by class
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[label].append(i)
        
        # Calculate likelihoods for each class
        for class_name in self.classes:
            indices = class_indices[class_name]
            
            # Sum feature counts for this class
            class_feature_counts = np.array(
                feature_matrix[indices].sum(axis=0)
            ).flatten()
            
            # Apply Laplace smoothing
            # P(feature|class) = (count + alpha) / (total_words + alpha * vocab_size)
            total_words = class_feature_counts.sum()
            smoothed_counts = class_feature_counts + self.alpha
            smoothed_total = total_words + self.alpha * self.vocabulary_size
            
            # Store log probabilities for numerical stability
            self.feature_likelihoods[class_name] = np.log(
                smoothed_counts / smoothed_total
            )
        
        self.logger.debug("Feature likelihoods calculated for all classes")
    
    def _store_training_stats(self, texts: List[str], labels: List[str]) -> None:
        """Store training statistics for model information."""
        label_counts = Counter(labels)
        
        self.training_stats = {
            'total_samples': len(texts),
            'num_classes': len(self.classes),
            'vocabulary_size': self.vocabulary_size,
            'alpha': self.alpha,
            'class_distribution': dict(label_counts),
            'avg_text_length': np.mean([len(text.split()) for text in texts]),
            'classes': self.classes.copy()
        }
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Return emotion probabilities for input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary mapping emotion names to probabilities
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. "
                           "Call train() first.")
        
        # Extract features for the input text
        feature_vector = self.preprocessor.transform_single(text)
        feature_array = feature_vector.toarray().flatten()
        
        # Calculate log probabilities for each class
        log_probs = {}
        for class_name in self.classes:
            # Log P(class) + sum(count * log P(feature|class))
            log_prob = np.log(self.class_priors[class_name])
            log_prob += np.sum(feature_array * self.feature_likelihoods[class_name])
            log_probs[class_name] = log_prob
        
        # Convert to probabilities using softmax for numerical stability
        max_log_prob = max(log_probs.values())
        exp_probs = {
            class_name: np.exp(log_prob - max_log_prob)
            for class_name, log_prob in log_probs.items()
        }
        
        # Normalize to get probabilities
        total_prob = sum(exp_probs.values())
        probabilities = {
            class_name: prob / total_prob
            for class_name, prob in exp_probs.items()
        }
        
        return probabilities
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """
        Return most likely emotion and confidence.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_emotion, confidence_score)
        """
        probabilities = self.predict(text)
        
        # Find the class with highest probability
        best_class = max(probabilities.items(), key=lambda x: x[1])
        
        return best_class[0], best_class[1]
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict emotions for multiple texts efficiently.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of probability dictionaries, one per input text
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # Extract features for all texts at once
        feature_matrix = self.preprocessor.extract_features(texts)
        
        results = []
        for i in range(len(texts)):
            feature_array = feature_matrix[i].toarray().flatten()
            
            # Calculate log probabilities for each class
            log_probs = {}
            for class_name in self.classes:
                log_prob = np.log(self.class_priors[class_name])
                log_prob += np.sum(feature_array * self.feature_likelihoods[class_name])
                log_probs[class_name] = log_prob
            
            # Convert to probabilities
            max_log_prob = max(log_probs.values())
            exp_probs = {
                class_name: np.exp(log_prob - max_log_prob)
                for class_name, log_prob in log_probs.items()
            }
            
            total_prob = sum(exp_probs.values())
            probabilities = {
                class_name: prob / total_prob
                for class_name, prob in exp_probs.items()
            }
            
            results.append(probabilities)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Serialize trained model to disk.
        
        Args:
            filepath: Path where to save the model
            
        Raises:
            ValueError: If model hasn't been trained yet
            IOError: If unable to save to the specified path
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model. Train the model first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare model data for serialization
            model_data = {
                'alpha': self.alpha,
                'class_priors': self.class_priors,
                'feature_likelihoods': {
                    class_name: likelihood.tolist()
                    for class_name, likelihood in self.feature_likelihoods.items()
                },
                'classes': self.classes,
                'vocabulary_size': self.vocabulary_size,
                'training_stats': self.training_stats,
                'preprocessor_config': self.preprocessor.get_preprocessing_info()
            }
            
            # Save the main model data as JSON for readability
            model_file = filepath.with_suffix('.json')
            with open(model_file, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            # Save the preprocessor separately (includes fitted vectorizer)
            preprocessor_file = filepath.with_suffix('.preprocessor.pkl')
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            self.logger.info(f"Model saved successfully to {model_file} and {preprocessor_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise IOError(f"Unable to save model to {filepath}: {e}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Raises:
            FileNotFoundError: If model files don't exist
            IOError: If unable to load the model
        """
        filepath = Path(filepath)
        model_file = filepath.with_suffix('.json')
        preprocessor_file = filepath.with_suffix('.preprocessor.pkl')
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        if not preprocessor_file.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_file}")
        
        try:
            # Load main model data
            with open(model_file, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            # Restore model parameters
            self.alpha = model_data['alpha']
            self.class_priors = model_data['class_priors']
            self.classes = model_data['classes']
            self.vocabulary_size = model_data['vocabulary_size']
            self.training_stats = model_data['training_stats']
            
            # Convert feature likelihoods back to numpy arrays
            self.feature_likelihoods = {
                class_name: np.array(likelihood)
                for class_name, likelihood in model_data['feature_likelihoods'].items()
            }
            
            # Load preprocessor
            with open(preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.is_trained = True
            self.logger.info(f"Model loaded successfully from {model_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise IOError(f"Unable to load model from {filepath}: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model configuration and training statistics
        """
        if not self.is_trained:
            return {
                'is_trained': False,
                'alpha': self.alpha
            }
        
        return {
            'is_trained': True,
            'alpha': self.alpha,
            'classes': self.classes.copy(),
            'num_classes': len(self.classes),
            'vocabulary_size': self.vocabulary_size,
            'training_stats': self.training_stats.copy(),
            'preprocessor_info': self.preprocessor.get_preprocessing_info()
        }
    
    def get_feature_importance(self, class_name: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get the most important features (words) for a specific emotion class.
        
        Args:
            class_name: Name of the emotion class
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance
            
        Raises:
            ValueError: If model isn't trained or class doesn't exist
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance.")
        
        if class_name not in self.classes:
            raise ValueError(f"Class '{class_name}' not found. Available classes: {self.classes}")
        
        # Get feature names and likelihoods for the class
        feature_names = self.preprocessor.get_feature_names()
        likelihoods = self.feature_likelihoods[class_name]
        
        # Create list of (feature, likelihood) pairs
        feature_importance = list(zip(feature_names, likelihoods))
        
        # Sort by likelihood (higher is more important)
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]