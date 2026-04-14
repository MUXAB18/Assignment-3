"""
Unit tests for TextPreprocessor class.
"""

import unittest
import numpy as np
from scipy.sparse import csr_matrix
from src.ml.text_processor import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextPreprocessor(remove_stopwords=True)
        self.processor_no_stopwords = TextPreprocessor(remove_stopwords=False)
    
    def test_normalize_text(self):
        """Test text normalization functionality."""
        # Test basic normalization
        text = "Hello, World! How are you doing today?"
        expected = "hello world how are you doing today"
        result = self.processor.normalize_text(text)
        self.assertEqual(result, expected)
        
        # Test with numbers and special characters
        text = "I'm feeling 100% great!!! @#$%"
        expected = "i m feeling 100 great"
        result = self.processor.normalize_text(text)
        self.assertEqual(result, expected)
        
        # Test with empty string
        result = self.processor.normalize_text("")
        self.assertEqual(result, "")
    
    def test_tokenize_with_stopwords(self):
        """Test tokenization with stop word removal."""
        text = "i am feeling very happy today"
        tokens = self.processor.tokenize(text)
        
        # Should remove stop words like 'i', 'am', 'very'
        self.assertNotIn('i', tokens)
        self.assertNotIn('am', tokens)
        self.assertNotIn('very', tokens)
        self.assertIn('feeling', tokens)
        self.assertIn('happy', tokens)
        self.assertIn('today', tokens)
    
    def test_tokenize_without_stopwords(self):
        """Test tokenization without stop word removal."""
        text = "i am feeling very happy today"
        tokens = self.processor_no_stopwords.tokenize(text)
        
        # Should keep all words
        expected_tokens = ['am', 'feeling', 'very', 'happy', 'today']
        for token in expected_tokens:
            self.assertIn(token, tokens)
    
    def test_preprocess_pipeline(self):
        """Test full preprocessing pipeline."""
        text = "I'm REALLY excited about this! It's amazing!!!"
        result = self.processor.preprocess(text)
        
        # Should be normalized, tokenized, and rejoined
        self.assertIsInstance(result, str)
        self.assertNotIn('!', result)
        self.assertNotIn("'", result)
        # Should contain meaningful words
        self.assertIn('really', result)
        self.assertIn('excited', result)
        self.assertIn('amazing', result)
    
    def test_fit_vocabulary(self):
        """Test vocabulary fitting."""
        texts = [
            "I am happy today",
            "I feel sad and lonely", 
            "This is amazing and wonderful",
            "I am angry about this situation"
        ]
        
        self.processor.fit_vocabulary(texts)
        
        # Should be fitted
        self.assertTrue(self.processor.is_fitted)
        self.assertIsNotNone(self.processor.vectorizer)
        self.assertGreater(self.processor.get_vocabulary_size(), 0)
    
    def test_extract_features(self):
        """Test feature extraction."""
        # First fit vocabulary
        train_texts = [
            "I am happy",
            "I feel sad",
            "This is amazing"
        ]
        self.processor.fit_vocabulary(train_texts)
        
        # Extract features
        test_texts = ["I am happy", "I feel sad"]
        features = self.processor.extract_features(test_texts)
        
        # Should return sparse matrix
        self.assertIsInstance(features, csr_matrix)
        self.assertEqual(features.shape[0], len(test_texts))
        self.assertEqual(features.shape[1], self.processor.get_vocabulary_size())
    
    def test_extract_features_without_fitting(self):
        """Test that feature extraction fails without fitting vocabulary."""
        with self.assertRaises(ValueError):
            self.processor.extract_features(["test text"])
    
    def test_transform_single(self):
        """Test single text transformation."""
        # Fit vocabulary first
        train_texts = ["I am happy", "I feel sad"]
        self.processor.fit_vocabulary(train_texts)
        
        # Transform single text
        result = self.processor.transform_single("I am happy")
        
        self.assertIsInstance(result, csr_matrix)
        self.assertEqual(result.shape[0], 1)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        train_texts = ["happy sad", "angry excited"]
        self.processor.fit_vocabulary(train_texts)
        
        feature_names = self.processor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        # Should contain some of our words
        self.assertTrue(any(word in feature_names for word in ['happy', 'sad', 'angry', 'excited']))
    
    def test_preprocessing_info(self):
        """Test getting preprocessing information."""
        info = self.processor.get_preprocessing_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('remove_stopwords', info)
        self.assertIn('min_word_freq', info)
        self.assertIn('max_features', info)
        self.assertIn('vocabulary_size', info)
        self.assertIn('is_fitted', info)
        
        # Before fitting
        self.assertEqual(info['vocabulary_size'], 0)
        self.assertFalse(info['is_fitted'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty text
        result = self.processor.preprocess("")
        self.assertEqual(result, "")
        
        # Non-string input
        result = self.processor.normalize_text(123)
        self.assertEqual(result, "123")
        
        # Very short text
        result = self.processor.preprocess("a")
        self.assertEqual(result, "")  # Single character should be filtered out


if __name__ == '__main__':
    unittest.main()