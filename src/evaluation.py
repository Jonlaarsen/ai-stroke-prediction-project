"""
Model evaluation module for stroke prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)


class ModelEvaluator:
    """
    Handles comprehensive model evaluation and visualization.
    
    Attributes:
        metrics (dict): Dictionary storing evaluation metrics
        feature_names (list): List of feature names for visualization
    """
    
    def __init__(self, feature_names=None):
        """
        Initialize ModelEvaluator.
        
        Args:
            feature_names (list): List of feature names for visualization
        """
        self.feature_names = feature_names
        self.metrics = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Compute comprehensive evaluation metrics for a model.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Name of the model for storing metrics
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store metrics
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return self.metrics[model_name]
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name='Model', ax=None):
        """
        Plot confusion matrix.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
            ax (matplotlib.axes.Axes): Optional axes to plot on
            
        Returns:
            matplotlib.axes.Axes: Axes with confusion matrix plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'],
            ax=ax
        )
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return ax
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name='Model', ax=None):
        """
        Plot ROC curve and compute AUC.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            ax (matplotlib.axes.Axes): Optional axes to plot on
            
        Returns:
            tuple: (fpr, tpr, auc_score) - ROC curve data and AUC score
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fpr, tpr, auc_score
    
    def plot_feature_importance(self, model, model_name='Model', top_n=10, ax=None):
        """
        Plot feature importance for Random Forest model.
        
        Args:
            model: Trained Random Forest model
            model_name (str): Name of the model
            top_n (int): Number of top features to display
            ax (matplotlib.axes.Axes): Optional axes to plot on
            
        Returns:
            matplotlib.axes.Axes: Axes with feature importance plot
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        if self.feature_names is None:
            self.feature_names = [f'Feature {i}' for i in range(len(model.feature_importances_))]
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        return ax
    
    def compare_models(self, model_names=None):
        """
        Compare metrics between multiple models.
        
        Args:
            model_names (list): List of model names to compare. If None, compares all stored models.
            
        Returns:
            pd.DataFrame: DataFrame with comparison metrics
        """
        import pandas as pd
        
        if model_names is None:
            model_names = list(self.metrics.keys())
        
        comparison_data = []
        for name in model_names:
            if name in self.metrics:
                metrics = self.metrics[name]
                comparison_data.append({
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics['roc_auc']
                })
        
        return pd.DataFrame(comparison_data)
    
    def print_metrics(self, model_name='Model'):
        """
        Print evaluation metrics for a model.
        
        Args:
            model_name (str): Name of the model
        """
        if model_name not in self.metrics:
            print(f"No metrics found for {model_name}")
            return
        
        metrics = self.metrics[model_name]
        print(f"\n{'='*50}")
        print(f"Evaluation Metrics - {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"{'='*50}\n")
