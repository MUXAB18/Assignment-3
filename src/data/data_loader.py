"""
Data loading and validation module for emotion classification system.

This module provides the DataLoader class for handling parquet file loading
from both split and unsplit directories, with comprehensive data validation.
"""

import os
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path


class DataLoader:
    """
    Handles loading and validation of emotion classification datasets.
    
    Supports loading from both split directories (train/validation/test) and
    unsplit directories, with comprehensive data validation to ensure datasets
    contain required columns and proper format.
    """
    
    def __init__(self, split_dir: str = "split", unsplit_dir: str = "unsplit"):
        """
        Initialize DataLoader with dataset directory paths.
        
        Args:
            split_dir: Path to directory containing split datasets
            unsplit_dir: Path to directory containing unsplit dataset
        """
        self.split_dir = Path(split_dir)
        self.unsplit_dir = Path(unsplit_dir)
        self.logger = logging.getLogger(__name__)
        
        # Expected columns in the dataset
        self.required_columns = {'text', 'label'}
        
        # Emotion label mapping (0-5 to emotion names)
        self.emotion_mapping = {
            0: 'sadness',
            1: 'joy', 
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
    
    def load_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test datasets from split directory.
        
        Returns:
            Tuple of (train_df, validation_df, test_df) DataFrames
            
        Raises:
            FileNotFoundError: If any required split files are missing
            ValueError: If datasets fail validation
        """
        self.logger.info(f"Loading split datasets from {self.split_dir}")
        
        # Define expected file patterns
        train_file = self._find_parquet_file(self.split_dir, "train")
        validation_file = self._find_parquet_file(self.split_dir, "validation")
        test_file = self._find_parquet_file(self.split_dir, "test")
        
        try:
            # Load datasets
            train_df = pd.read_parquet(train_file)
            validation_df = pd.read_parquet(validation_file)
            test_df = pd.read_parquet(test_file)
            
            # Validate each dataset
            self._validate_dataset(train_df, "train")
            self._validate_dataset(validation_df, "validation") 
            self._validate_dataset(test_df, "test")
            
            # Add emotion column for easier interpretation
            train_df = self._add_emotion_column(train_df)
            validation_df = self._add_emotion_column(validation_df)
            test_df = self._add_emotion_column(test_df)
            
            self.logger.info(f"Successfully loaded split datasets: "
                           f"train={len(train_df)}, val={len(validation_df)}, test={len(test_df)}")
            
            return train_df, validation_df, test_df
            
        except Exception as e:
            self.logger.error(f"Failed to load split datasets: {e}")
            raise
    
    def load_unsplit_data(self) -> pd.DataFrame:
        """
        Load training data from unsplit directory.
        
        Returns:
            DataFrame containing the unsplit training dataset
            
        Raises:
            FileNotFoundError: If unsplit file is missing
            ValueError: If dataset fails validation
        """
        self.logger.info(f"Loading unsplit dataset from {self.unsplit_dir}")
        
        train_file = self._find_parquet_file(self.unsplit_dir, "train")
        
        try:
            # Load dataset
            df = pd.read_parquet(train_file)
            
            # Validate dataset
            self._validate_dataset(df, "unsplit")
            
            # Add emotion column for easier interpretation
            df = self._add_emotion_column(df)
            
            self.logger.info(f"Successfully loaded unsplit dataset: {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load unsplit dataset: {e}")
            raise
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """
        Public interface for dataset validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if dataset is valid
            
        Raises:
            ValueError: If dataset fails validation
        """
        return self._validate_dataset(df, "external")
    
    def get_emotion_labels(self, df: pd.DataFrame) -> List[str]:
        """
        Extract unique emotion categories from dataset.
        
        Args:
            df: DataFrame containing emotion data
            
        Returns:
            List of unique emotion label strings
        """
        if 'emotion' in df.columns:
            return sorted(df['emotion'].unique().tolist())
        elif 'label' in df.columns:
            unique_labels = sorted(df['label'].unique())
            return [self.emotion_mapping.get(label, f"unknown_{label}") 
                   for label in unique_labels]
        else:
            raise ValueError("Dataset must contain 'emotion' or 'label' column")
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        info = {
            'total_samples': len(df),
            'columns': df.columns.tolist(),
            'emotion_distribution': {},
            'text_length_stats': {},
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Emotion distribution
        if 'emotion' in df.columns:
            info['emotion_distribution'] = df['emotion'].value_counts().to_dict()
        elif 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            info['emotion_distribution'] = {
                self.emotion_mapping.get(label, f"unknown_{label}"): count
                for label, count in label_counts.items()
            }
        
        # Text length statistics
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            info['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'median': float(text_lengths.median()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': float(text_lengths.std())
            }
        
        return info
    
    def check_data_availability(self) -> Dict[str, bool]:
        """
        Check availability of split and unsplit datasets.
        
        Returns:
            Dictionary indicating which datasets are available
        """
        availability = {
            'split_available': False,
            'unsplit_available': False,
            'split_files': {},
            'unsplit_files': {}
        }
        
        # Check split files
        try:
            train_file = self._find_parquet_file(self.split_dir, "train")
            val_file = self._find_parquet_file(self.split_dir, "validation")
            test_file = self._find_parquet_file(self.split_dir, "test")
            
            availability['split_available'] = True
            availability['split_files'] = {
                'train': str(train_file),
                'validation': str(val_file),
                'test': str(test_file)
            }
        except FileNotFoundError:
            pass
        
        # Check unsplit files
        try:
            train_file = self._find_parquet_file(self.unsplit_dir, "train")
            availability['unsplit_available'] = True
            availability['unsplit_files'] = {
                'train': str(train_file)
            }
        except FileNotFoundError:
            pass
        
        return availability
    
    def _find_parquet_file(self, directory: Path, prefix: str) -> Path:
        """
        Find parquet file with given prefix in directory.
        
        Args:
            directory: Directory to search in
            prefix: File prefix to match
            
        Returns:
            Path to the found parquet file
            
        Raises:
            FileNotFoundError: If no matching file is found
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        
        # Look for files matching pattern: {prefix}-*.parquet
        pattern = f"{prefix}-*.parquet"
        matching_files = list(directory.glob(pattern))
        
        if not matching_files:
            raise FileNotFoundError(f"No parquet files found matching '{pattern}' in {directory}")
        
        if len(matching_files) > 1:
            self.logger.warning(f"Multiple files found matching '{pattern}', using first: {matching_files[0]}")
        
        return matching_files[0]
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """
        Validate dataset format and structure.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of dataset for error messages
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(f"Dataset '{dataset_name}' is empty")
        
        # Check required columns
        missing_columns = self.required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Dataset '{dataset_name}' missing required columns: {missing_columns}")
        
        # Check for null values in required columns
        for col in self.required_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(f"Dataset '{dataset_name}' has {null_count} null values in column '{col}'")
        
        # Validate text column (can be 'object' or 'str' dtype)
        if not (df['text'].dtype == 'object' or df['text'].dtype.name == 'str' or df['text'].dtype.name == 'string'):
            raise ValueError(f"Dataset '{dataset_name}' text column must be string type, got {df['text'].dtype}")
        
        # Check for empty text entries
        empty_text_count = (df['text'].str.strip() == '').sum()
        if empty_text_count > 0:
            self.logger.warning(f"Dataset '{dataset_name}' has {empty_text_count} empty text entries")
        
        # Validate label column
        if not pd.api.types.is_numeric_dtype(df['label']):
            raise ValueError(f"Dataset '{dataset_name}' label column must be numeric")
        
        # Check label range (should be 0-5 for emotion classification)
        valid_labels = set(self.emotion_mapping.keys())
        dataset_labels = set(df['label'].unique())
        invalid_labels = dataset_labels - valid_labels
        
        if invalid_labels:
            raise ValueError(f"Dataset '{dataset_name}' contains invalid labels: {invalid_labels}. "
                           f"Valid labels are: {valid_labels}")
        
        self.logger.info(f"Dataset '{dataset_name}' validation passed: {len(df)} samples, "
                        f"{len(dataset_labels)} emotion classes")
        
        return True
    
    def _add_emotion_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable emotion column based on numeric labels.
        
        Args:
            df: DataFrame with numeric labels
            
        Returns:
            DataFrame with added 'emotion' column
        """
        df = df.copy()
        df['emotion'] = df['label'].map(self.emotion_mapping)
        return df