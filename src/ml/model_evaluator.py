"""
Model evaluation module for emotion classification system.

This module provides the ModelEvaluator class that calculates comprehensive
performance metrics for the emotion classifier, including accuracy, precision,
recall, F1-score, and confusion matrices.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
import logging

from .emotion_classifier import EmotionClassifier


class ModelEvaluator:
    """
    Evaluates emotion classification model performance and manages metrics storage.
    
    This class provides comprehensive evaluation capabilities including:
    - Overall accuracy calculation
    - Per-class precision, recall, and F1-score
    - Confusion matrix generation
    - Historical metrics tracking
    """
    
    def __init__(self, classifier: EmotionClassifier, metrics_db_path: str = "data/metrics.db"):
        """
        Initialize ModelEvaluator with a trained classifier.
        
        Args:
            classifier: Trained EmotionClassifier instance
            metrics_db_path: Path to SQLite database for storing metrics
        """
        self.classifier = classifier
        self.metrics_db_path = Path(metrics_db_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure metrics database directory exists
        self.metrics_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                # Create model_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT NOT NULL,
                        overall_accuracy REAL NOT NULL,
                        evaluation_timestamp DATETIME NOT NULL,
                        total_samples INTEGER NOT NULL,
                        metrics_json TEXT NOT NULL
                    )
                """)
                
                # Create class_metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS class_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_metrics_id INTEGER REFERENCES model_metrics(id),
                        emotion_class TEXT NOT NULL,
                        precision REAL NOT NULL,
                        recall REAL NOT NULL,
                        f1_score REAL NOT NULL
                    )
                """)
                
                conn.commit()
                self.logger.info("Metrics database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics database: {e}")
            raise
    
    def evaluate(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for the model.
        
        Args:
            test_texts: List of test text samples
            test_labels: List of corresponding true emotion labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if not test_texts or not test_labels:
            raise ValueError("Test texts and labels cannot be empty")
        
        if len(test_texts) != len(test_labels):
            raise ValueError("Number of test texts must match number of labels")
        
        self.logger.info(f"Evaluating model on {len(test_texts)} test samples")
        
        # Get predictions for all test samples
        predictions = []
        for text in test_texts:
            pred_emotion, _ = self.classifier.predict_single(text)
            predictions.append(pred_emotion)
        
        # Calculate overall accuracy
        correct_predictions = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
        overall_accuracy = correct_predictions / len(test_labels)
        
        # Calculate per-class metrics
        per_class_metrics = self.per_class_metrics(test_texts, test_labels)
        
        # Generate confusion matrix
        confusion_matrix = self.confusion_matrix(test_texts, test_labels)
        
        # Compile results
        evaluation_results = {
            "overall_accuracy": overall_accuracy,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "total_samples": len(test_texts),
            "evaluation_timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "true_labels": test_labels
        }
        
        self.logger.info(f"Model evaluation completed. Overall accuracy: {overall_accuracy:.4f}")
        
        return evaluation_results
    
    def confusion_matrix(self, test_texts: List[str], test_labels: List[str]) -> np.ndarray:
        """
        Generate confusion matrix for visualization.
        
        Args:
            test_texts: List of test text samples
            test_labels: List of corresponding true emotion labels
            
        Returns:
            Confusion matrix as numpy array
        """
        # Get unique emotion classes
        unique_labels = sorted(set(test_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Initialize confusion matrix
        matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        
        # Fill confusion matrix
        for text, true_label in zip(test_texts, test_labels):
            pred_emotion, _ = self.classifier.predict_single(text)
            
            # Handle case where prediction is not in training labels
            if pred_emotion in label_to_idx:
                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[pred_emotion]
                matrix[true_idx][pred_idx] += 1
            else:
                # If predicted emotion is unknown, count as misclassification
                self.logger.warning(f"Unknown predicted emotion: {pred_emotion}")
        
        return matrix
    
    def per_class_metrics(self, test_texts: List[str], test_labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1-score for each emotion class.
        
        Args:
            test_texts: List of test text samples
            test_labels: List of corresponding true emotion labels
            
        Returns:
            Dictionary with per-class metrics
        """
        # Get predictions
        predictions = []
        for text in test_texts:
            pred_emotion, _ = self.classifier.predict_single(text)
            predictions.append(pred_emotion)
        
        # Get unique classes
        unique_labels = sorted(set(test_labels + predictions))
        
        # Calculate metrics for each class
        metrics = {}
        
        for emotion_class in unique_labels:
            # True positives, false positives, false negatives
            tp = sum(1 for pred, true in zip(predictions, test_labels) 
                    if pred == emotion_class and true == emotion_class)
            fp = sum(1 for pred, true in zip(predictions, test_labels) 
                    if pred == emotion_class and true != emotion_class)
            fn = sum(1 for pred, true in zip(predictions, test_labels) 
                    if pred != emotion_class and true == emotion_class)
            
            # Calculate precision, recall, F1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[emotion_class] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": sum(1 for label in test_labels if label == emotion_class)
            }
        
        return metrics
    
    def store_metrics(self, evaluation_results: Dict[str, Any], model_version: str = "1.0") -> int:
        """
        Store evaluation metrics in the database for historical tracking.
        
        Args:
            evaluation_results: Results from evaluate() method
            model_version: Version identifier for the model
            
        Returns:
            ID of the stored metrics record
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main metrics record
                cursor.execute("""
                    INSERT INTO model_metrics 
                    (model_version, overall_accuracy, evaluation_timestamp, total_samples, metrics_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model_version,
                    evaluation_results["overall_accuracy"],
                    evaluation_results["evaluation_timestamp"],
                    evaluation_results["total_samples"],
                    json.dumps(evaluation_results)
                ))
                
                metrics_id = cursor.lastrowid
                
                # Insert per-class metrics
                for emotion_class, class_metrics in evaluation_results["per_class_metrics"].items():
                    cursor.execute("""
                        INSERT INTO class_metrics 
                        (model_metrics_id, emotion_class, precision, recall, f1_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        metrics_id,
                        emotion_class,
                        class_metrics["precision"],
                        class_metrics["recall"],
                        class_metrics["f1_score"]
                    ))
                
                conn.commit()
                self.logger.info(f"Metrics stored successfully with ID: {metrics_id}")
                
                return metrics_id
                
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
            raise
    
    def get_historical_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve historical evaluation metrics from the database.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of historical metrics records
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, model_version, overall_accuracy, evaluation_timestamp, 
                           total_samples, metrics_json
                    FROM model_metrics
                    ORDER BY evaluation_timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                records = []
                for row in cursor.fetchall():
                    record = {
                        "id": row[0],
                        "model_version": row[1],
                        "overall_accuracy": row[2],
                        "evaluation_timestamp": row[3],
                        "total_samples": row[4],
                        "full_metrics": json.loads(row[5])
                    }
                    records.append(record)
                
                return records
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical metrics: {e}")
            return []
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent evaluation metrics.
        
        Returns:
            Latest metrics dictionary or None if no metrics exist
        """
        historical = self.get_historical_metrics(limit=1)
        return historical[0] if historical else None
    
    def compare_models(self, model_versions: List[str]) -> Dict[str, Any]:
        """
        Compare performance metrics across different model versions.
        
        Args:
            model_versions: List of model version identifiers to compare
            
        Returns:
            Comparison results with metrics for each version
        """
        try:
            with sqlite3.connect(self.metrics_db_path) as conn:
                cursor = conn.cursor()
                
                comparison = {}
                
                for version in model_versions:
                    cursor.execute("""
                        SELECT overall_accuracy, evaluation_timestamp, total_samples, metrics_json
                        FROM model_metrics
                        WHERE model_version = ?
                        ORDER BY evaluation_timestamp DESC
                        LIMIT 1
                    """, (version,))
                    
                    row = cursor.fetchone()
                    if row:
                        comparison[version] = {
                            "overall_accuracy": row[0],
                            "evaluation_timestamp": row[1],
                            "total_samples": row[2],
                            "full_metrics": json.loads(row[3])
                        }
                    else:
                        comparison[version] = None
                
                return comparison
                
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return {}
    
    def generate_performance_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            evaluation_results: Results from evaluate() method
            
        Returns:
            Formatted performance report as string
        """
        report_lines = [
            "=" * 60,
            "EMOTION CLASSIFICATION MODEL PERFORMANCE REPORT",
            "=" * 60,
            f"Evaluation Date: {evaluation_results['evaluation_timestamp']}",
            f"Total Test Samples: {evaluation_results['total_samples']}",
            f"Overall Accuracy: {evaluation_results['overall_accuracy']:.4f} ({evaluation_results['overall_accuracy']*100:.2f}%)",
            "",
            "PER-CLASS PERFORMANCE METRICS:",
            "-" * 40
        ]
        
        # Add per-class metrics
        for emotion_class, metrics in evaluation_results["per_class_metrics"].items():
            report_lines.extend([
                f"Class: {emotion_class}",
                f"  Precision: {metrics['precision']:.4f}",
                f"  Recall:    {metrics['recall']:.4f}",
                f"  F1-Score:  {metrics['f1_score']:.4f}",
                f"  Support:   {metrics['support']} samples",
                ""
            ])
        
        # Add confusion matrix info
        confusion_matrix = np.array(evaluation_results["confusion_matrix"])
        report_lines.extend([
            "CONFUSION MATRIX:",
            "-" * 20,
            f"Matrix Shape: {confusion_matrix.shape}",
            f"Diagonal Sum (Correct Predictions): {np.trace(confusion_matrix)}",
            ""
        ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)