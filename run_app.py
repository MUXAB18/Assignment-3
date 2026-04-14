#!/usr/bin/env python3
"""
Run script for the Emotion Classification System Flask application.
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.web_app import create_app
from utils.config import Config

def main():
    """Main function to run the Flask application."""
    # Load configuration
    config = Config()
    
    # Create Flask app
    app = create_app(config)
    
    print(f"Starting Emotion Classification System...")
    print(f"Server will be available at: http://{config.web.host}:{config.web.port}")
    print(f"Debug mode: {config.web.debug}")
    print("Press Ctrl+C to stop the server")
    
    # Run the application
    try:
        app.run(
            host=config.web.host,
            port=config.web.port,
            debug=config.web.debug
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()