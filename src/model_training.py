"""
Model training module for stroke prediction.
"""

import pickle

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    """
    Handles training and hyperparameter tuning of ML models.
    
    Attributes:
        logistic_model: Trained Logistic Regression model
        random_forest_model: Trained Random Forest model
        best_params_logistic: Best hyperparameters for Logistic Regression
        best_params_rf: Best hyperparameters for Random Forest
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.logistic_model = None
        self.random_forest_model = None
        self.best_params_logistic = None
        self.best_params_rf = None
        
    def train_logistic_regression(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train Logistic Regression model with optional hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            tune_hyperparameters (bool): Whether to perform GridSearchCV tuning
            
        Returns:
            LogisticRegression: Trained model
        """
        if tune_hyperparameters:
            # Define parameter grid
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            # Base model with balanced class weights
            base_model = LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.logistic_model = grid_search.best_estimator_
            self.best_params_logistic = grid_search.best_params_
            
            print(f"Best Logistic Regression parameters: {self.best_params_logistic}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        else:
            # Train without tuning
            self.logistic_model = LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000,
                C=1.0,
                penalty='l2',
                solver='liblinear'
            )
            self.logistic_model.fit(X_train, y_train)
        
        return self.logistic_model
    
    def train_random_forest(self, X_train, y_train, tune_hyperparameters=True):
        """
        Train Random Forest model with optional hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            tune_hyperparameters (bool): Whether to perform GridSearchCV tuning
            
        Returns:
            RandomForestClassifier: Trained model
        """
        if tune_hyperparameters:
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            }
            
            # Base model
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.random_forest_model = grid_search.best_estimator_
            self.best_params_rf = grid_search.best_params_
            
            print(f"Best Random Forest parameters: {self.best_params_rf}")
            print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        else:
            # Train without tuning
            self.random_forest_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            self.random_forest_model.fit(X_train, y_train)
        
        return self.random_forest_model
    
    def get_model(self, model_type='logistic'):
        """
        Get trained model by type.
        
        Args:
            model_type (str): 'logistic' or 'random_forest'
            
        Returns:
            Trained model or None if not trained
        """
        if model_type == 'logistic':
            return self.logistic_model
        elif model_type == 'random_forest':
            return self.random_forest_model
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
    
    def save_models(self, filepath_logistic='models/logistic_model.pkl',
                    filepath_rf='models/random_forest_model.pkl'):
        """
        Save trained models to disk.
        
        Args:
            filepath_logistic (str): Path to save Logistic Regression model
            filepath_rf (str): Path to save Random Forest model
        """
        import os
        os.makedirs(os.path.dirname(filepath_logistic), exist_ok=True)
        os.makedirs(os.path.dirname(filepath_rf), exist_ok=True)
        
        if self.logistic_model:
            with open(filepath_logistic, 'wb') as f:
                pickle.dump(self.logistic_model, f)
        
        if self.random_forest_model:
            with open(filepath_rf, 'wb') as f:
                pickle.dump(self.random_forest_model, f)
    
    def load_models(self, filepath_logistic='models/logistic_model.pkl',
                   filepath_rf='models/random_forest_model.pkl'):
        """
        Load trained models from disk.
        
        Args:
            filepath_logistic (str): Path to Logistic Regression model
            filepath_rf (str): Path to Random Forest model
        """
        try:
            with open(filepath_logistic, 'rb') as f:
                self.logistic_model = pickle.load(f)
        except FileNotFoundError:
            print(f"Logistic Regression model not found at {filepath_logistic}")
        
        try:
            with open(filepath_rf, 'rb') as f:
                self.random_forest_model = pickle.load(f)
        except FileNotFoundError:
            print(f"Random Forest model not found at {filepath_rf}")
