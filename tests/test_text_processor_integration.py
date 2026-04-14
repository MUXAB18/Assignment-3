"""
Integration test for TextPreprocessor with real dataset.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.text_processor import TextPreprocessor
from src.data.data_loader import DataLoader


class TestTextProcessorIntegration(unittest.TestCase):
    """Integration tests for TextPreprocessor with real data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TextPreprocessor()
        self.data_loader = DataLoader()
    
    def test_with_real_dataset(self):
        """Test TextPreprocessor with actual emotion dataset."""
        try:
            # Try to load split data first
            train_df, val_df, test_df = self.data_loader.load_split_data()
            sample_texts = train_df['text'].head(10).tolist()
        except Exception:
            try:
                # Fallback to unsplit data
                unsplit_df = self.data_loader.load_unsplit_data()
                sample_texts = unsplit_df['text'].head(10).tolist()
            except Exception as e:
                self.skipTest(f"Could not load dataset: {e}")
        
        # Test preprocessing on real data
        processed_texts = []
        for text in sample_texts:
            processed = self.processor.preprocess(text)
            processed_texts.append(processed)
            
            # Verify preprocessing worked
            self.assertIsInstance(processed, str)
            # Should not contain punctuation
            self.assertNotIn('!', processed)
            self.assertNotIn('?', processed)
            self.assertNotIn(',', processed)
        
        # Test vocabulary fitting
        self.processor.fit_vocabulary(sample_texts)
        self.assertTrue(self.processor.is_fitted)
        self.assertGreater(self.processor.get_vocabulary_size(), 0)
        
        # Test feature extraction
        features = self.processor.extract_features(sample_texts)
        self.assertEqual(features.shape[0], len(sample_texts))
        self.assertEqual(features.shape[1], self.processor.get_vocabulary_size())
        
        print(f"Successfully processed {len(sample_texts)} texts")
        print(f"Vocabulary size: {self.processor.get_vocabulary_size()}")
        print(f"Feature matrix shape: {features.shape}")
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent between training and prediction."""
        sample_texts = [
            "I am feeling really happy today!",
            "This makes me so angry and frustrated.",
            "I feel sad and lonely right now."
        ]
        
        # Fit vocabulary
        self.processor.fit_vocabulary(sample_texts)
        
        # Extract features for training texts
        train_features = self.processor.extract_features(sample_texts)
        
        # Extract features for same texts (simulating prediction)
        pred_features = self.processor.extract_features(sample_texts)
        
        # Should be identical
        self.assertTrue((train_features != pred_features).nnz == 0)
        
        # Test with new text that uses same vocabulary
        new_text = ["I am happy"]
        new_features = self.processor.extract_features(new_text)
        self.assertEqual(new_features.shape[1], train_features.shape[1])


if __name__ == '__main__':
    unittest.main()