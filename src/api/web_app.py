"""
Flask web application for the Emotion Classification System.

This module provides the web interface for users to input text and receive
emotion predictions, as well as view model performance metrics.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from typing import Dict, Any, Optional
import logging
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.prediction_engine import PredictionEngine
from ml.model_evaluator import ModelEvaluator
from ml.emotion_classifier import EmotionClassifier
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config: Optional[Config] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__, 
                template_folder='../../templates',
                static_folder='../../static')
    
    # Load configuration
    if config is None:
        config = Config()
    
    app.config['SECRET_KEY'] = config.web.secret_key
    app.config['DEBUG'] = config.web.debug
    
    # Initialize prediction engine
    try:
        prediction_engine = PredictionEngine()
        app.prediction_engine = prediction_engine
        logger.info("Prediction engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction engine: {e}")
        app.prediction_engine = None
    
    # Initialize model evaluator for metrics
    try:
        if app.prediction_engine and app.prediction_engine.classifier:
            evaluator = ModelEvaluator(app.prediction_engine.classifier)
            app.model_evaluator = evaluator
            logger.info("Model evaluator initialized successfully")
        else:
            app.model_evaluator = None
    except Exception as e:
        logger.error(f"Failed to initialize model evaluator: {e}")
        app.model_evaluator = None
    
    @app.route('/')
    def index():
        """Main page with text input form."""
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Handle emotion prediction requests."""
        try:
            # Get text from form or JSON
            if request.is_json:
                data = request.get_json()
                text = data.get('text', '').strip()
            else:
                text = request.form.get('text', '').strip()
            
            # Validate input
            if not text:
                error_msg = "Please enter some text to analyze."
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('index.html', error=error_msg)
            
            if len(text) > 5000:  # Reasonable limit
                error_msg = "Text is too long. Please limit to 5000 characters."
                if request.is_json:
                    return jsonify({'error': error_msg}), 400
                return render_template('index.html', error=error_msg, text=text)
            
            # Check if prediction engine is available
            if not app.prediction_engine:
                error_msg = "Prediction service is currently unavailable. Please try again later."
                if request.is_json:
                    return jsonify({'error': error_msg}), 503
                return render_template('index.html', error=error_msg, text=text)
            
            # Make prediction
            result = app.prediction_engine.predict_emotion(text)
            
            # Return JSON for API calls
            if request.is_json:
                return jsonify(result)
            
            # Render results page for web interface
            return render_template('results.html', 
                                 text=text, 
                                 result=result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            error_msg = "An error occurred while processing your request. Please try again."
            
            if request.is_json:
                return jsonify({'error': error_msg}), 500
            return render_template('index.html', error=error_msg, text=request.form.get('text', ''))
    
    @app.route('/metrics')
    def metrics():
        """Display model performance metrics."""
        try:
            if not app.model_evaluator:
                return render_template('metrics.html',
                                       error="Model metrics are currently unavailable.")

            # Load test data and evaluate
            import sys, os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from data.data_loader import DataLoader

            data_loader = DataLoader()
            try:
                _, _, test_df = data_loader.load_split_data()
                test_texts = test_df['text'].tolist()
                test_labels = test_df['emotion'].tolist()
            except Exception:
                unsplit_df = data_loader.load_unsplit_data()
                sample = unsplit_df.sample(n=min(500, len(unsplit_df)), random_state=42)
                test_texts = sample['text'].tolist()
                test_labels = sample['emotion'].tolist()

            # Check for cached metrics first
            latest = app.model_evaluator.get_latest_metrics()
            if latest:
                metrics_data = latest['full_metrics']
            else:
                results = app.model_evaluator.evaluate(test_texts, test_labels)
                app.model_evaluator.store_metrics(results, "1.0")
                metrics_data = results

            return render_template('metrics.html', metrics=metrics_data)

        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return render_template('metrics.html',
                                   error=f"An error occurred while loading metrics: {str(e)}")
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """API endpoint for emotion prediction."""
        return predict()
    
    @app.route('/api/metrics', methods=['GET'])
    def api_metrics():
        """API endpoint for model metrics."""
        try:
            if not app.model_evaluator:
                return jsonify({'error': 'Model metrics unavailable'}), 503

            latest = app.model_evaluator.get_latest_metrics()
            if latest:
                return jsonify(latest['full_metrics'])

            return jsonify({'status': 'No metrics available yet. Visit /metrics to generate them.'})

        except Exception as e:
            logger.error(f"API metrics error: {e}")
            return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', 
                             error_code=404,
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html',
                             error_code=500,
                             error_message="Internal server error"), 500
    
    return app

def main():
    """Run the Flask application."""
    app = create_app()
    
    # Get configuration
    config = Config()
    
    app.run(
        host=config.web.host,
        port=config.web.port,
        debug=config.web.debug
    )

if __name__ == '__main__':
    main()