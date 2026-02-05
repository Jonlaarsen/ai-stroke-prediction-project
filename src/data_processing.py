"""
Data processing module for stroke prediction dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class DataProcessor:
    """
    Handles all data preprocessing steps for the stroke prediction dataset.
    
    Attributes:
        scaler (StandardScaler): Scaler for numerical features
        imputer_numerical (SimpleImputer): Imputer for numerical features
        imputer_categorical (SimpleImputer): Imputer for categorical features
        label_encoders (dict): Dictionary of label encoders for categorical columns
        categorical_columns (list): List of categorical column names
        numerical_columns (list): List of numerical column names
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize DataProcessor.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer_numerical = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, file_path):
        """
        Load dataset from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        df = pd.read_csv(file_path)
        return df
    
    def identify_columns(self, df):
        """
        Identify categorical and numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Common categorical columns in stroke dataset
        potential_categorical = ['gender', 'ever_married', 'work_type', 
                                'Residence_type', 'residence_type', 'smoking_status']
        
        self.categorical_columns = [col for col in potential_categorical 
                                   if col in df.columns]
        
        # Also check for any remaining object/string columns
        for col in df.columns:
            if col not in self.categorical_columns and col != 'stroke' and col != 'id':
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    self.categorical_columns.append(col)
        
        # Numerical columns are all others except target and id
        self.numerical_columns = [col for col in df.columns 
                                 if col not in self.categorical_columns 
                                 and col != 'stroke' and col != 'id']
        
    def handle_missing_values(self, df, fit=True):
        """
        Handle missing values using median for numerical and most_frequent for categorical.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit imputers (True for training, False for prediction)
            
        Returns:
            pd.DataFrame: Dataframe with imputed missing values
        """
        df_processed = df.copy()
        
        # Handle numerical columns
        if self.numerical_columns:
            if fit:
                df_processed[self.numerical_columns] = self.imputer_numerical.fit_transform(
                    df_processed[self.numerical_columns]
                )
            else:
                df_processed[self.numerical_columns] = self.imputer_numerical.transform(
                    df_processed[self.numerical_columns]
                )
        
        # Handle categorical columns
        if self.categorical_columns:
            if fit:
                df_processed[self.categorical_columns] = self.imputer_categorical.fit_transform(
                    df_processed[self.categorical_columns]
                )
            else:
                df_processed[self.categorical_columns] = self.imputer_categorical.transform(
                    df_processed[self.categorical_columns]
                )
        
        return df_processed
    
    def encode_categorical(self, df, fit=True):
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit encoders (True for training, False for prediction)
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            if fit:
                df_encoded[col] = self.label_encoders[col].fit_transform(
                    df_encoded[col].astype(str)
                )
            else:
                # For prediction, use transform only
                df_encoded[col] = self.label_encoders[col].transform(
                    df_encoded[col].astype(str)
                )
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix
            fit (bool): Whether to fit scaler (True for training, False for prediction)
            
        Returns:
            np.ndarray: Scaled feature matrix
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def prepare_data(self, df):
        """
        Complete data preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Split and processed data
        """
        # Identify column types
        self.identify_columns(df)
        
        # Handle missing values (fit=True for training)
        df_processed = self.handle_missing_values(df, fit=True)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_processed, fit=True)
        
        # Separate features and target
        X = df_encoded.drop(columns=['stroke', 'id'] if 'id' in df_encoded.columns else ['stroke'])
        y = df_encoded['stroke']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train[self.numerical_columns], fit=True)
        X_test_scaled = self.scale_features(X_test[self.numerical_columns], fit=False)
        
        # Combine scaled numerical and encoded categorical features
        X_train_final = np.hstack([
            X_train_scaled,
            X_train[self.categorical_columns].values
        ])
        
        X_test_final = np.hstack([
            X_test_scaled,
            X_test[self.categorical_columns].values
        ])
        
        # Store feature names for later use
        self.feature_names = self.numerical_columns + self.categorical_columns
        
        return X_train_final, X_test_final, y_train.values, y_test.values
    
    def is_fitted(self):
        """
        Check if the DataProcessor has been fitted (scaler and imputers are ready).
        
        Returns:
            bool: True if fitted, False otherwise
        """
        try:
            # Check if scaler has been fitted by checking if it has mean_ attribute
            has_scaler = hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None
            # Check if imputers have been fitted
            has_num_imputer = (hasattr(self.imputer_numerical, 'statistics_') and 
                              self.imputer_numerical.statistics_ is not None)
            has_cat_imputer = (hasattr(self.imputer_categorical, 'statistics_') and 
                              self.imputer_categorical.statistics_ is not None)
            # Check if we have column information
            has_columns = len(self.numerical_columns) > 0 or len(self.categorical_columns) > 0
            return has_scaler and has_num_imputer and has_cat_imputer and has_columns
        except:
            return False
    
    def prepare_single_prediction(self, input_dict):
        """
        Prepare a single data point for prediction.
        
        Args:
            input_dict (dict): Dictionary with feature values
            
        Returns:
            np.ndarray: Processed feature array ready for prediction
            
        Raises:
            ValueError: If DataProcessor is not fitted
        """
        if not self.is_fitted():
            raise ValueError(
                "DataProcessor is not fitted. Please call prepare_data() first "
                "or ensure the scaler and imputers have been fitted."
            )
        
        # Create dataframe from input
        df = pd.DataFrame([input_dict])
        
        # Handle missing values (fit=False for prediction, shouldn't be needed for user input, but safe)
        df_processed = self.handle_missing_values(df, fit=False)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical(df_processed, fit=False)
        
        # Extract features in correct order
        numerical_features = df_encoded[self.numerical_columns].values
        categorical_features = df_encoded[self.categorical_columns].values
        
        # Scale numerical features
        numerical_scaled = self.scale_features(numerical_features, fit=False)
        
        # Combine
        feature_array = np.hstack([numerical_scaled, categorical_features])
        
        return feature_array
