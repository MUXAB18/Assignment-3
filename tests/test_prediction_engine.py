"""
Unit tests for PredictionEngine class.

Tests prediction accuracy, response format, batch prediction functionality,
edge case handling, and response time requirements.
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from src.ml.prediction_engine import PredictionEngine
from src.ml.emotion_classifier import EmotionClassifier


class TestPredictionEngine(unittest.TestCase):
    """Test cases for PredictionEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the classifier to avoid loading actual models in tests
        self.mock_classifier = Mock(spec=EmotionClassifier)
        self.mock_classifier.is_trained = True
        self.mock_classifier.classes = ["joy", "sadness", "anger", "fear"]
        self.mock_classifier.alpha = 1.0
        
        # Sample prediction results
        self.sample_predictions = {
            "joy": 0.7,
            "sadness": 0.15,
            "anger": 0.1,
            "fear": 0.05
        }
    
    @patch('src.ml.prediction_engine.EmotionClassifier')
    @patch('pathlib.Path.exists')
    def test_init_with_default_model(self, mock_exists, mock_classifier_class):
        """Test initialization with default model path."""
        mock_exists.return_value = True
        mock_classifier_instance = Mock()
        mock_classifier_class.return_value = mock_classifier_instance
        
        engine = PredictionEngine()
        
        self.assertEqual(engine.model_path, "data/models/real_data_model")
        mock_classifier_instance.load_model.assert_called_once_with("data/models/real_data_model")
    
    @patch('src.ml.prediction_engine.EmotionClassifier')
    @patch('pathlib.Path.exists')
    def test_init_with_custom_model(self, mock_exists, mock_classifier_class):
        """Test initialization with custom model path."""
        mock_exists.return_value = True
        mock_classifier_instance = Mock()
        mock_classifier_class.return_value = mock_classifier_instance
        
        custom_path = "data/models/custom_model"
        engine = PredictionEngine(model_path=custom_path)
        
        self.assertEqual(engine.model_path, custom_path)
        mock_classifier_instance.load_model.assert_called_once_with(custom_path)
    
    @patch('src.ml.prediction_engine.EmotionClassifier')
    @patch('pathlib.Path.exists')
    def test_init_fallback_to_test_model(self, mock_exists, mock_classifier_class):
        """Test fallback to test model when primary model doesn't exist."""
        # Primary model doesn't exist, test model exists
        mock_exists.return_value = False  # First call (primary model)
        mock_exists.side_effect = [False, True]  # Primary false, test true
        mock_classifier_instance = Mock()
        mock_classifier_class.return_value = mock_classifier_instance
        
        engine = PredictionEngine()
        
        self.assertEqual(engine.model_path, "data/models/test_emotion_model")
        mock_classifier_instance.load_model.assert_called_once_with("data/models/test_emotion_model")
    
    @patch('src.ml.prediction_engine.EmotionClassifier')
    @patch('pathlib.Path.exists')
    def test_init_no_model_found(self, mock_exists, mock_classifier_class):
        """Test initialization when no model is found."""
        mock_exists.return_value = False
        mock_classifier_class.return_value = Mock()
        
        with self.assertRaises(FileNotFoundError):
            PredictionEngine()
    
    def test_predict_emotion_success(self):
        """Test successful emotion prediction."""
        # Create engine with mocked classifier
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        # Mock the predict method
        self.mock_classifier.predict.return_value = self.sample_predictions
        
        result = engine.predict_emotion("I am very happy today!")
        
        # Verify result structure
        self.assertIn("emotion", result)
        self.assertIn("confidence", result)
        self.assertIn("all_predictions", result)
        self.assertIn("processing_time", result)
        self.assertIn("text_length", result)
        self.assertIn("is_valid", result)
        
        # Verify values
        self.assertEqual(result["emotion"], "joy")
        self.assertEqual(result["confidence"], 0.7)
        self.assertEqual(result["all_predictions"], self.sample_predictions)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["text_length"], 22)  # "I am very happy today!" is 22 characters
        self.assertIsInstance(result["processing_time"], float)
        self.assertGreater(result["processing_time"], 0)
    
    def test_predict_emotion_empty_text(self):
        """Test prediction with empty text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        result = engine.predict_emotion("")
        
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["is_valid"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Empty or whitespace-only text")
    
    def test_predict_emotion_whitespace_only(self):
        """Test prediction with whitespace-only text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        result = engine.predict_emotion("   \n\t  ")
        
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["confidence"], 0.0)
        self.assertFalse(result["is_valid"])
        self.assertIn("error", result)
    
    def test_predict_emotion_very_short_text(self):
        """Test prediction with very short text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        result = engine.predict_emotion("Hi")
        
        self.assertEqual(result["emotion"], "neutral")
        self.assertEqual(result["confidence"], 0.1)
        self.assertFalse(result["is_valid"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Text too short for reliable classification")
    
    def test_predict_emotion_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.is_loaded = False
        
        with self.assertRaises(ValueError) as context:
            engine.predict_emotion("Test text")
        
        self.assertIn("Model is not loaded", str(context.exception))
    
    def test_batch_predict_success(self):
        """Test successful batch prediction."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        texts = ["I am happy", "I am sad", "I am angry"]
        batch_results = [
            {"joy": 0.8, "sadness": 0.1, "anger": 0.05, "fear": 0.05},
            {"joy": 0.1, "sadness": 0.8, "anger": 0.05, "fear": 0.05},
            {"joy": 0.05, "sadness": 0.1, "anger": 0.8, "fear": 0.05}
        ]
        
        self.mock_classifier.predict_batch.return_value = batch_results
        
        results = engine.batch_predict(texts)
        
        self.assertEqual(len(results), 3)
        
        # Check first result
        self.assertEqual(results[0]["emotion"], "joy")
        self.assertEqual(results[0]["confidence"], 0.8)
        self.assertTrue(results[0]["is_valid"])
        
        # Check second result
        self.assertEqual(results[1]["emotion"], "sadness")
        self.assertEqual(results[1]["confidence"], 0.8)
        
        # Check third result
        self.assertEqual(results[2]["emotion"], "anger")
        self.assertEqual(results[2]["confidence"], 0.8)
    
    def test_batch_predict_with_edge_cases(self):
        """Test batch prediction with mixed valid and invalid texts."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        texts = ["I am happy", "", "Hi", "I am very sad today"]
        valid_batch_results = [
            {"joy": 0.8, "sadness": 0.1, "anger": 0.05, "fear": 0.05},
            {"joy": 0.1, "sadness": 0.8, "anger": 0.05, "fear": 0.05}
        ]
        
        self.mock_classifier.predict_batch.return_value = valid_batch_results
        
        results = engine.batch_predict(texts)
        
        self.assertEqual(len(results), 4)
        
        # Valid text
        self.assertTrue(results[0]["is_valid"])
        self.assertEqual(results[0]["emotion"], "joy")
        
        # Empty text
        self.assertFalse(results[1]["is_valid"])
        self.assertEqual(results[1]["emotion"], "neutral")
        
        # Short text
        self.assertFalse(results[2]["is_valid"])
        self.assertEqual(results[2]["emotion"], "neutral")
        
        # Valid text
        self.assertTrue(results[3]["is_valid"])
        self.assertEqual(results[3]["emotion"], "sadness")
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.is_loaded = True
        
        with self.assertRaises(ValueError) as context:
            engine.batch_predict([])
        
        self.assertIn("Texts list cannot be empty", str(context.exception))
    
    def test_batch_predict_model_not_loaded(self):
        """Test batch prediction when model is not loaded."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.is_loaded = False
        
        with self.assertRaises(ValueError) as context:
            engine.batch_predict(["Test text"])
        
        self.assertIn("Model is not loaded", str(context.exception))
    
    def test_get_model_info_loaded(self):
        """Test getting model info when model is loaded."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.model_path = "test/model/path"
        
        with patch('pathlib.Path.exists', return_value=True):
            info = engine.get_model_info()
        
        self.assertEqual(info["model_path"], "test/model/path")
        self.assertTrue(info["is_loaded"])
        self.assertTrue(info["model_exists"])
        self.assertTrue(info["is_trained"])
        self.assertEqual(info["classes"], ["joy", "sadness", "anger", "fear"])
        self.assertEqual(info["num_classes"], 4)
        self.assertEqual(info["alpha"], 1.0)
    
    def test_get_model_info_not_loaded(self):
        """Test getting model info when model is not loaded."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.is_loaded = False
        engine.model_path = "test/model/path"
        
        with patch('pathlib.Path.exists', return_value=False):
            info = engine.get_model_info()
        
        self.assertEqual(info["model_path"], "test/model/path")
        self.assertFalse(info["is_loaded"])
        self.assertFalse(info["model_exists"])
    
    def test_validate_input_valid(self):
        """Test input validation with valid text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        
        is_valid, error = engine.validate_input("This is a valid text for prediction")
        
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_input_not_string(self):
        """Test input validation with non-string input."""
        engine = PredictionEngine.__new__(PredictionEngine)
        
        is_valid, error = engine.validate_input(123)
        
        self.assertFalse(is_valid)
        self.assertEqual(error, "Input must be a string")
    
    def test_validate_input_empty(self):
        """Test input validation with empty text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        
        is_valid, error = engine.validate_input("")
        
        self.assertFalse(is_valid)
        self.assertIn("cannot be empty", error)
    
    def test_validate_input_too_short(self):
        """Test input validation with too short text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        
        is_valid, error = engine.validate_input("Hi")
        
        self.assertFalse(is_valid)
        self.assertIn("too short", error)
    
    def test_validate_input_too_long(self):
        """Test input validation with too long text."""
        engine = PredictionEngine.__new__(PredictionEngine)
        
        long_text = "a" * 10001
        is_valid, error = engine.validate_input(long_text)
        
        self.assertFalse(is_valid)
        self.assertIn("too long", error)
    
    def test_response_time_requirement(self):
        """Test that prediction meets response time requirement."""
        engine = PredictionEngine.__new__(PredictionEngine)
        engine.classifier = self.mock_classifier
        engine.is_loaded = True
        engine.logger = Mock()
        
        # Mock predict to simulate processing time
        def mock_predict(text):
            time.sleep(0.001)  # Simulate small processing time
            return self.sample_predictions
        
        self.mock_classifier.predict.side_effect = mock_predict
        
        result = engine.predict_emotion("This is a test text for timing")
        
        # Should complete well under 2 seconds
        self.assertLess(result["processing_time"], 2.0)
        self.assertGreater(result["processing_time"], 0)


if __name__ == '__main__':
    unittest.main()