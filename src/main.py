"""
Main script for training and evaluating stroke prediction models.

"""

import sys

import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
import matplotlib.pyplot as plt


def main():
    """Main function to run the complete ML pipeline."""
    
    print("="*60)
    print("Stroke Prediction - Machine Learning Pipeline")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    data_processor = DataProcessor(test_size=0.2, random_state=42)
    model_trainer = ModelTrainer(random_state=42)
    model_evaluator = ModelEvaluator()
    
    # Load data
    print("\n2. Loading data...")
    data_path = 'data/stroke.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please download the stroke dataset and place it in the data/ directory.")
        return
    
    df = data_processor.load_data(data_path)
    print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Prepare data
    print("\n3. Preprocessing data...")
    X_train, X_test, y_train, y_test = data_processor.prepare_data(df)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set feature names for evaluator
    model_evaluator.feature_names = data_processor.feature_names
    
    # Train Logistic Regression
    print("\n4. Training Logistic Regression model...")
    logistic_model = model_trainer.train_logistic_regression(
        X_train, y_train, tune_hyperparameters=True
    )
    
    # Evaluate Logistic Regression
    print("\n5. Evaluating Logistic Regression model...")
    model_evaluator.evaluate_model(
        logistic_model, X_test, y_test, model_name='Logistic Regression'
    )
    model_evaluator.print_metrics('Logistic Regression')
    
    # Train Random Forest
    print("\n6. Training Random Forest model...")
    rf_model = model_trainer.train_random_forest(
        X_train, y_train, tune_hyperparameters=True
    )
    
    # Evaluate Random Forest
    print("\n7. Evaluating Random Forest model...")
    model_evaluator.evaluate_model(
        rf_model, X_test, y_test, model_name='Random Forest'
    )
    model_evaluator.print_metrics('Random Forest')
    
    # Model comparison
    print("\n8. Model Comparison:")
    comparison_df = model_evaluator.compare_models()
    print(comparison_df.to_string(index=False))
    
    # Save models
    print("\n9. Saving models...")
    os.makedirs('models', exist_ok=True)
    model_trainer.save_models()
    print("Models saved successfully!")
    
    # Create visualizations
    print("\n10. Creating visualizations...")
    os.makedirs('results', exist_ok=True)
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metrics_lr = model_evaluator.metrics['Logistic Regression']
    metrics_rf = model_evaluator.metrics['Random Forest']
    
    model_evaluator.plot_confusion_matrix(
        metrics_lr['y_test'], metrics_lr['y_pred'],
        'Logistic Regression', ax=axes[0]
    )
    model_evaluator.plot_confusion_matrix(
        metrics_rf['y_test'], metrics_rf['y_pred'],
        'Random Forest', ax=axes[1]
    )
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: results/confusion_matrices.png")
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    model_evaluator.plot_roc_curve(
        metrics_lr['y_test'], metrics_lr['y_pred_proba'],
        'Logistic Regression', ax=ax
    )
    model_evaluator.plot_roc_curve(
        metrics_rf['y_test'], metrics_rf['y_pred_proba'],
        'Random Forest', ax=ax
    )
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: results/roc_curves.png")
    
    # Feature importance (Random Forest)
    fig, ax = plt.subplots(figsize=(10, 8))
    model_evaluator.plot_feature_importance(
        rf_model, 'Random Forest', top_n=10, ax=ax
    )
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/feature_importance.png")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nTo run the Streamlit app, use:")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
