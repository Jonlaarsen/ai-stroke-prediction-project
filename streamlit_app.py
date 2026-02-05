"""
Standalone Streamlit application entry point.

Run with: streamlit run streamlit_app.py
"""

import sys

import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st

import numpy as np

from data_processing import DataProcessor

from model_training import ModelTrainer

from evaluation import ModelEvaluator

from app import StreamlitApp


def main():
    """Initialize and run Streamlit app."""
    # Check if models exist
    models_dir = 'models'
    logistic_path = os.path.join(models_dir, 'logistic_model.pkl')
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    
    if not os.path.exists(logistic_path) or not os.path.exists(rf_path):
        st.warning(" Models not found. Please train models first by running: `python src/main.py`")
        st.info("The app will still work, but predictions require trained models.")
    
    # Initialize components
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    model_evaluator = ModelEvaluator()
    
    # Try to load models if they exist
    try:
        model_trainer.load_models()
        
        # Load data to set up data processor (needed for feature names, encoders, and scaler)
        data_path = 'data/stroke.csv'
        if os.path.exists(data_path):
            df = data_processor.load_data(data_path)
            
            # Use prepare_data to properly fit all transformers (scaler, imputers, encoders)
            # This ensures the DataProcessor is ready for predictions
            X_train, X_test, y_train, y_test = data_processor.prepare_data(df)
            
            # Set feature names for evaluator
            model_evaluator.feature_names = data_processor.feature_names
            
            # If models are loaded, evaluate them
            if model_trainer.logistic_model and model_trainer.random_forest_model:
                # Evaluate models
                model_evaluator.evaluate_model(
                    model_trainer.logistic_model, X_test, y_test,
                    model_name='Logistic Regression'
                )
                model_evaluator.evaluate_model(
                    model_trainer.random_forest_model, X_test, y_test,
                    model_name='Random Forest'
                )
    except Exception as e:
        st.warning(f"Could not load models or evaluate: {str(e)}")
        import traceback
        st.error(f"Error details: {traceback.format_exc()}")
    
    # Create and run app
    app = StreamlitApp(data_processor, model_trainer, model_evaluator)
    app.run()


if __name__ == "__main__":
    main()
