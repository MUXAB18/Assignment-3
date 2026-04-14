"""
Unit tests for ModelEvaluator class.

Tests the evaluation metrics calculation, confusion matrix generation,
and metrics storage functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import sqlite3

from src.ml.emotion_classifier import EmotionClassifier
from src.ml.model_evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_metrics.db"
        
        # Create and train a simple classifier
        self.classifier = EmotionClassifier(alpha=1.0)
        
        # Simple training data
        train_texts = [
            "I am so happy today",
            "This is wonderful news",
            "I feel sad and lonely", 
            "This makes me angry",
            "I am excited about this",
            "I feel terrible about this"
        ]
        train_labels = ["joy", "joy", "sadness", "anger", "joy", "sadness"]
        
        self.classifier.train(train_texts, train_labels)
        
        # Create evaluator
        self.evaluator = ModelEvaluator(self.classifier, str(self.db_path))
        
        # Test data
        self.test_texts = [
            "I am very happy",
            "This is sad news", 
            "I am furious",
            "Great job everyone"
        ]
        self.test_labels = ["joy", "sadness", "anger", "joy"]
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_evaluate_basic_functionality(self):
        """Test basic evaluation functionality."""
        results = self.evaluator.evaluate(self.test_texts, self.test_labels)
        
        # Check required keys exist
        self.assertIn("overall_accuracy", results)
        self.assertIn("per_class_metrics", results)
        self.assertIn("confusion_matrix", results)
        self.assertIn("total_samples", results)
        self.assertIn("evaluation_timestamp", results)
        
        # Check data types and ranges
        self.assertIsInstance(results["overall_accuracy"], float)
        self.assertGreaterEqual(results["overall_accuracy"], 0.0)
        self.assertLessEqual(results["overall_accuracy"], 1.0)
        self.assertEqual(results["total_samples"], len(self.test_texts))
        
        # Check confusion matrix shape
        confusion_matrix = np.array(results["confusion_matrix"])
        self.assertEqual(len(confusion_matrix.shape), 2)
        self.assertEqual(confusion_matrix.shape[0], confusion_matrix.shape[1])
    
    def test_per_class_metrics(self):
        """Test per-class metrics calculation."""
        metrics = self.evaluator.per_class_metrics(self.test_texts, self.test_labels)
        
        # Check that metrics exist for each class
        for emotion_class in set(self.test_labels):
            self.assertIn(emotion_class, metrics)
            
            class_metrics = metrics[emotion_class]
            self.assertIn("precision", class_metrics)
            self.assertIn("recall", class_metrics)
            self.assertIn("f1_score", class_metrics)
            self.assertIn("support", class_metrics)
            
            # Check value ranges
            self.assertGreaterEqual(class_metrics["precision"], 0.0)
            self.assertLessEqual(class_metrics["precision"], 1.0)
            self.assertGreaterEqual(class_metrics["recall"], 0.0)
            self.assertLessEqual(class_metrics["recall"], 1.0)
            self.assertGreaterEqual(class_metrics["f1_score"], 0.0)
            self.assertLessEqual(class_metrics["f1_score"], 1.0)
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        matrix = self.evaluator.confusion_matrix(self.test_texts, self.test_labels)
        
        # Check matrix properties
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(len(matrix.shape), 2)
        self.assertEqual(matrix.shape[0], matrix.shape[1])
        
        # Check that matrix contains non-negative integers
        self.assertTrue(np.all(matrix >= 0))
        self.assertTrue(np.issubdtype(matrix.dtype, np.integer))
        
        # Check that sum equals number of test samples
        self.assertEqual(np.sum(matrix), len(self.test_texts))
    
    def test_store_and_retrieve_metrics(self):
        """Test metrics storage and retrieval."""
        # Evaluate and store metrics
        results = self.evaluator.evaluate(self.test_texts, self.test_labels)
        metrics_id = self.evaluator.store_metrics(results, "test_v1.0")
        
        # Check that ID was returned
        self.assertIsInstance(metrics_id, int)
        self.assertGreater(metrics_id, 0)
        
        # Retrieve historical metrics
        historical = self.evaluator.get_historical_metrics(limit=5)
        self.assertGreater(len(historical), 0)
        
        # Check latest metrics
        latest = self.evaluator.get_latest_metrics()
        self.assertIsNotNone(latest)
        self.assertEqual(latest["model_version"], "test_v1.0")
    
    def test_database_initialization(self):
        """Test that database is properly initialized."""
        # Check that database file exists
        self.assertTrue(self.db_path.exists())
        
        # Check that tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check model_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_metrics'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check class_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='class_metrics'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_generate_performance_report(self):
        """Test performance report generation."""
        results = self.evaluator.evaluate(self.test_texts, self.test_labels)
        report = self.evaluator.generate_performance_report(results)
        
        # Check that report is a string and contains expected content
        self.assertIsInstance(report, str)
        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("Overall Accuracy", report)
        self.assertIn("PER-CLASS PERFORMANCE", report)
        self.assertIn("CONFUSION MATRIX", report)
    
    def test_empty_input_validation(self):
        """Test validation of empty inputs."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([], [])
        
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(["test"], [])
        
        with self.assertRaises(ValueError):
            self.evaluator.evaluate([], ["joy"])
    
    def test_mismatched_input_validation(self):
        """Test validation of mismatched input lengths."""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(["text1", "text2"], ["joy"])


if __name__ == "__main__":
    unittest.main()