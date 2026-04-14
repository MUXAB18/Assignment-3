"""
Configuration settings for the emotion classification system.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Naive Bayes model."""
    alpha: float = 1.0              # Laplace smoothing parameter
    remove_stopwords: bool = True   # Text preprocessing option
    min_word_freq: int = 2          # Minimum word frequency threshold
    max_features: int = 10000       # Maximum vocabulary size
    model_version: str = "1.0"      # Model version identifier


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    split_dir: str = "split"
    unsplit_dir: str = "unsplit"
    models_dir: str = "data/models"
    text_column: str = "text"
    emotion_column: str = "label"


@dataclass
class WebConfig:
    """Configuration for the Flask web application."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    secret_key: str = "dev-secret-key-change-in-production"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    log_file: Optional[str] = "logs/emotion_classifier.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.web = WebConfig()
        self.logging = LoggingConfig()
        
        # Override with environment variables if present
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model config
        if os.getenv('MODEL_ALPHA'):
            self.model.alpha = float(os.getenv('MODEL_ALPHA'))
        if os.getenv('MODEL_REMOVE_STOPWORDS'):
            self.model.remove_stopwords = os.getenv('MODEL_REMOVE_STOPWORDS').lower() == 'true'
        
        # Web config
        if os.getenv('FLASK_HOST'):
            self.web.host = os.getenv('FLASK_HOST')
        if os.getenv('FLASK_PORT'):
            self.web.port = int(os.getenv('FLASK_PORT'))
        if os.getenv('FLASK_DEBUG'):
            self.web.debug = os.getenv('FLASK_DEBUG').lower() == 'true'
        if os.getenv('FLASK_SECRET_KEY'):
            self.web.secret_key = os.getenv('FLASK_SECRET_KEY')
        
        # Logging config
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.logging.log_file = os.getenv('LOG_FILE')


# Global configuration instance
config = Config()