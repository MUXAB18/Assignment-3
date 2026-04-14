"""
Prediction Engine for Emotion Classification System.

This module provides the main interface for making emotion predictions,
orchestrating text preprocessing and model inference with comprehensive
error handling and performance monitoring.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

from .emotion_classifier import EmotionClassifier
from .text_processor import TextPreprocessor


class PredictionEngine:
    """
    Orchestrates text preprocessing and model inference for emotion classification.
    
    Provides single text prediction with confidence scores, batch prediction
    capability for multiple texts, and handles edge cases like empty text
    and very short text inputs.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize PredictionEngine with optional pre-trained model.
        
        Args:
            model_path: Path to trained model file (without extension).
                       If None, uses default model path.
        """
        self.logger = logging.getLogger(__name__)
        self.classifier = EmotionClassifier()
        self.model_path = model_path or "data/models/real_data_model"
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and preprocessor."""
        try:
            model_file = Path(f"{self.model_path}.json")
            if not model_file.exists():
                # Try test model as fallback
                fallback_path = "data/models/test_emotion_model"
                fallback_file = Path(f"{fallback_path}.json")
                if fallback_file.exists():
                    self.model_path = fallback_path
                    self.logger.warning(f"Primary model not found, using fallback: {fallback_path}")
                else:
                    raise FileNotFoundError(f"No trained model found at {self.model_path} or fallback")
            
            self.classifier.load_model(self.model_path)
            self.is_loaded = True
            self.logger.info(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise
    
    def predict_emotion(self, text: str) -> Dict[str, Any]:
        """
        Main prediction interface returning structured results.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing:
                - emotion: Most likely emotion
                - confidence: Confidence score for predicted emotion
                - all_predictions: All emotion probabilities
                - processing_time: Time taken for prediction (seconds)
                - text_length: Length of input text
                - is_valid: Whether input was valid for prediction
                
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If prediction fails
        """
        if not self.is_loaded:
            raise ValueError("Model is not loaded. Cannot make predictions.")
        
        start_time = time.time()
        
        try:
            # Handle edge cases
            if not text or not text.strip():
                return self._create_empty_result(text, time.time() - start_time)
            
            # Handle very short text (less than 3 characters)
            if len(text.strip()) < 3:
                return self._create_short_text_result(text, time.time() - start_time)
            
            # Get all emotion probabilities
            all_probabilities = self.classifier.predict(text)
            
            # Find the most likely emotion
            predicted_emotion = max(all_probabilities.items(), key=lambda x: x[1])
            
            processing_time = time.time() - start_time
            
            result = {
                "emotion": predicted_emotion[0],
                "confidence": predicted_emotion[1],
                "all_predictions": all_probabilities,
                "processing_time": processing_time,
                "text_length": len(text),
                "is_valid": True
            }
            
            # Log if processing time exceeds requirement (2 seconds)
            if processing_time > 2.0:
                self.logger.warning(f"Prediction took {processing_time:.3f}s, exceeding 2s requirement")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Prediction failed for text length {len(text)}: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts efficiently.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction dictionaries, one per input text
            
        Raises:
            ValueError: If model is not loaded or texts list is empty
        """
        if not self.is_loaded:
            raise ValueError("Model is not loaded. Cannot make predictions.")
        
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        start_time = time.time()
        results = []
        
        try:
            # Separate valid texts from edge cases
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip() and len(text.strip()) >= 3:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            # Get batch predictions for valid texts
            if valid_texts:
                batch_probabilities = self.classifier.predict_batch(valid_texts)
            else:
                batch_probabilities = []
            
            # Process all texts, handling edge cases
            valid_idx = 0
            for i, text in enumerate(texts):
                text_start_time = time.time()
                
                if i in valid_indices:
                    # Use batch prediction result
                    all_probabilities = batch_probabilities[valid_idx]
                    predicted_emotion = max(all_probabilities.items(), key=lambda x: x[1])
                    
                    result = {
                        "emotion": predicted_emotion[0],
                        "confidence": predicted_emotion[1],
                        "all_predictions": all_probabilities,
                        "processing_time": time.time() - text_start_time,
                        "text_length": len(text),
                        "is_valid": True
                    }
                    valid_idx += 1
                else:
                    # Handle edge cases
                    if not text or not text.strip():
                        result = self._create_empty_result(text, time.time() - text_start_time)
                    else:
                        result = self._create_short_text_result(text, time.time() - text_start_time)
                
                results.append(result)
            
            total_time = time.time() - start_time
            self.logger.info(f"Batch prediction completed: {len(texts)} texts in {total_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")
    
    def _create_empty_result(self, text: str, processing_time: float) -> Dict[str, Any]:
        """Create result for empty or whitespace-only text."""
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_predictions": {"neutral": 1.0},
            "processing_time": processing_time,
            "text_length": len(text),
            "is_valid": False,
            "error": "Empty or whitespace-only text"
        }
    
    def _create_short_text_result(self, text: str, processing_time: float) -> Dict[str, Any]:
        """Create result for very short text (less than 3 characters)."""
        return {
            "emotion": "neutral",
            "confidence": 0.1,
            "all_predictions": {"neutral": 1.0},
            "processing_time": processing_time,
            "text_length": len(text),
            "is_valid": False,
            "error": "Text too short for reliable classification"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata and status.
        
        Returns:
            Dictionary containing model information
        """
        try:
            info = {
                "model_path": self.model_path,
                "is_loaded": self.is_loaded,
                "model_exists": Path(f"{self.model_path}.json").exists()
            }
            
            if self.is_loaded:
                info.update({
                    "is_trained": self.classifier.is_trained,
                    "classes": list(self.classifier.classes) if self.classifier.classes else [],
                    "num_classes": len(self.classifier.classes) if self.classifier.classes else 0,
                    "alpha": self.classifier.alpha
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {
                "model_path": self.model_path,
                "is_loaded": False,
                "error": str(e)
            }
    
    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """
        Reload the model, optionally from a different path.
        
        Args:
            model_path: New model path (without extension). If None, reloads current model.
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if model_path:
                self.model_path = model_path
            
            self.is_loaded = False
            self._load_model()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload model: {e}")
            return False
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Validate input text for prediction.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if not text or not text.strip():
            return False, "Input text cannot be empty or whitespace-only"
        
        if len(text.strip()) < 3:
            return False, "Input text too short for reliable classification (minimum 3 characters)"
        
        if len(text) > 10000:  # Reasonable upper limit
            return False, "Input text too long (maximum 10,000 characters)"
        
        return True, ""