"""
Streamlit application for stroke prediction.
"""
import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from data_processing import DataProcessor

from model_training import ModelTrainer

from evaluation import ModelEvaluator


class StreamlitApp:
    """
    Streamlit application for interactive stroke prediction.
    
    This class handles the Streamlit UI and integrates with
    DataProcessor, ModelTrainer, and ModelEvaluator.
    """
    
    def __init__(self, data_processor, model_trainer, model_evaluator):
        """
        Initialize StreamlitApp.
        """
        self.data_processor = data_processor
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        
        # Set page config
        st.set_page_config(
            page_title="Stroke Prediction App",
            layout="wide"
        )
    
    def run(self):
        """Run the Streamlit application."""
        st.title("Stroke Prediction Application")
        st.markdown("---")
        
        # Sidebar for model selection
        st.sidebar.header("Model Selection")
        compare_models = st.sidebar.checkbox(
            "Compare Both Models",
            value=True,
            help="Show predictions from both models side-by-side"
        )
        
        if not compare_models:
            model_choice = st.sidebar.selectbox(
                "Choose Model",
                ["Logistic Regression", "Random Forest"],
                help="Select the ML model for prediction"
            )
        else:
            model_choice = None  # Will use both models
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "Prediction", 
            "Model Performance", 
            "Feature Importance",
            "About"
        ])
        
        with tab1:
            self._prediction_tab(model_choice, compare_models)
        
        with tab2:
            self._performance_tab(model_choice, compare_models)
        
        with tab3:
            self._feature_importance_tab(model_choice)
        
        with tab4:
            self._about_tab()
    
    def _prediction_tab(self, model_choice, compare_models):
        """Prediction tab with user input form."""
        st.header("Make a Prediction")
        st.markdown("Enter patient information to predict stroke risk.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographic Information")
            age = st.slider("Age", 0, 100, 50, help="Patient age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Patient gender")
            ever_married = st.selectbox("Ever Married", ["Yes", "No"], help="Marital status")
            work_type = st.selectbox(
                "Work Type",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                help="Type of work"
            )
            residence_type = st.selectbox(
                "Residence Type",
                ["Urban", "Rural"],
                help="Type of residence"
            )
        
        with col2:
            st.subheader("Medical Information")
            avg_glucose_level = st.slider(
                "Average Glucose Level",
                50.0, 300.0, 95.0, 0.1,
                help="Average glucose level in blood"
            )
            bmi = st.slider(
                "BMI",
                10.0, 50.0, 25.0, 0.1,
                help="Body Mass Index"
            )
            smoking_status = st.selectbox(
                "Smoking Status",
                ["formerly smoked", "never smoked", "smokes", "Unknown"],
                help="Smoking habits"
            )
            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="History of hypertension"
            )
            heart_disease = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="History of heart disease"
            )
        
        # Prediction button
        if st.button("Predict Stroke Risk", type="primary", use_container_width=True):
            # Prepare input dictionary
            input_dict = {
                'age': age,
                'gender': gender,
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status,
                'hypertension': hypertension,
                'heart_disease': heart_disease
            }
            
            try:
                # Check if DataProcessor is fitted
                if not self.data_processor.is_fitted():
                    st.error("DataProcessor is not initialized. Please train models first by running: `python src/main.py`")
                    return
                
                # Prepare data for prediction
                X_input = self.data_processor.prepare_single_prediction(input_dict)
                
                # Get models and make predictions
                if compare_models:
                    # Compare both models
                    logistic_model = self.model_trainer.get_model('logistic')
                    rf_model = self.model_trainer.get_model('random_forest')
                    
                    if logistic_model is None or rf_model is None:
                        st.error("Both models must be trained. Please run: `python src/main.py`")
                        return
                    
                    # Get predictions from both models
                    lr_prediction = logistic_model.predict(X_input)[0]
                    lr_probability = logistic_model.predict_proba(X_input)[0][1]
                    
                    rf_prediction = rf_model.predict(X_input)[0]
                    rf_probability = rf_model.predict_proba(X_input)[0][1]
                    
                    # Display comparison
                    st.markdown("---")
                    st.subheader("Prediction Results - Model Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Logistic Regression")
                        if lr_prediction == 1:
                            st.error(f"**Stroke Risk Detected**")
                        else:
                            st.success(f"**No Stroke Risk**")
                        st.metric("Risk Probability", f"{lr_probability * 100:.2f}%")
                        
                        # Risk interpretation
                        if lr_probability < 0.3:
                            st.info("Low Risk")
                        elif lr_probability < 0.7:
                            st.warning("Moderate Risk")
                        else:
                            st.error("High Risk")
                    
                    with col2:
                        st.markdown("### Random Forest")
                        if rf_prediction == 1:
                            st.error(f"**Stroke Risk Detected**")
                        else:
                            st.success(f"**No Stroke Risk**")
                        st.metric("Risk Probability", f"{rf_probability * 100:.2f}%")
                        
                        # Risk interpretation
                        if rf_probability < 0.3:
                            st.info("Low Risk")
                        elif rf_probability < 0.7:
                            st.warning("Moderate Risk")
                        else:
                            st.error("High Risk")
                    
                    # Show difference
                    st.markdown("---")
                    prob_diff = abs(lr_probability - rf_probability)
                    avg_prob = (lr_probability + rf_probability) / 2
                    
                    diff_col1, diff_col2 = st.columns(2)
                    with diff_col1:
                        st.metric("Probability Difference", f"{prob_diff * 100:.2f} pp", 
                                 help="Absolute difference between model predictions")
                    with diff_col2:
                        st.metric("Average Probability", f"{avg_prob * 100:.2f}%",
                                 help="Consensus estimate from both models")
                    
                    if lr_prediction != rf_prediction:
                        st.warning("**Models Disagree**: Different algorithms may give different predictions. Check the Performance tab to see which model performs better.")
                    else:
                        st.success("**Models Agree**: Both models predict the same outcome.")
                else:
                    # Single model prediction
                    model = self.model_trainer.get_model(
                        'logistic' if model_choice == "Logistic Regression" else 'random_forest'
                    )
                    
                    if model is None:
                        st.error(f"{model_choice} model not trained yet. Please train the model first.")
                        return
                    
                    # Make prediction
                    prediction = model.predict(X_input)[0]
                    probability = model.predict_proba(X_input)[0][1]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if prediction == 1:
                            st.error(f"**Stroke Risk Detected**")
                        else:
                            st.success(f"**No Stroke Risk**")
                    
                    with result_col2:
                        st.metric("Risk Probability", f"{probability * 100:.2f}%")
                    
                    # Risk interpretation
                    st.markdown("---")
                    if probability < 0.3:
                        st.info("**Low Risk**: The model predicts a low probability of stroke.")
                    elif probability < 0.7:
                        st.warning("**Moderate Risk**: The model predicts a moderate probability of stroke. Consider consulting a healthcare professional.")
                    else:
                        st.error("**High Risk**: The model predicts a high probability of stroke. Please consult a healthcare professional immediately.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
    
    def _performance_tab(self, model_choice, compare_models):
        """Model performance tab with metrics and visualizations."""
        st.header("Model Performance Metrics")
        
        if compare_models or model_choice is None:
            # Show both models
            if len(self.model_evaluator.metrics) == 0:
                st.warning("No model metrics available. Please train and evaluate models first.")
                return
            
            st.subheader("Model Comparison")
            comparison_df = self.model_evaluator.compare_models()
            st.dataframe(comparison_df, use_container_width=True)
            
            # Show visualizations for both
            if 'Logistic Regression' in self.model_evaluator.metrics and 'Random Forest' in self.model_evaluator.metrics:
                st.markdown("---")
                st.subheader("Confusion Matrices")
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                lr_metrics = self.model_evaluator.metrics['Logistic Regression']
                rf_metrics = self.model_evaluator.metrics['Random Forest']
                
                self.model_evaluator.plot_confusion_matrix(
                    lr_metrics['y_test'], lr_metrics['y_pred'],
                    'Logistic Regression', ax=axes[0]
                )
                self.model_evaluator.plot_confusion_matrix(
                    rf_metrics['y_test'], rf_metrics['y_pred'],
                    'Random Forest', ax=axes[1]
                )
                st.pyplot(fig)
                
                st.markdown("---")
                st.subheader("ROC Curves Comparison")
                fig, ax = plt.subplots(figsize=(10, 8))
                self.model_evaluator.plot_roc_curve(
                    lr_metrics['y_test'], lr_metrics['y_pred_proba'],
                    'Logistic Regression', ax=ax
                )
                self.model_evaluator.plot_roc_curve(
                    rf_metrics['y_test'], rf_metrics['y_pred_proba'],
                    'Random Forest', ax=ax
                )
                st.pyplot(fig)
            return
        
        model_type = 'logistic' if model_choice == "Logistic Regression" else 'random_forest'
        model = self.model_trainer.get_model(model_type)
        
        if model is None:
            st.warning(f"{model_choice} model not trained yet. Please train the model first.")
            return
        
        # Check if metrics exist
        if model_choice not in self.model_evaluator.metrics:
            st.info("Model metrics not available. Please run evaluation first.")
            return
        
        metrics = self.model_evaluator.metrics[model_choice]
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        # Visualizations
        st.markdown("---")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            self.model_evaluator.plot_confusion_matrix(
                metrics['y_test'],
                metrics['y_pred'],
                model_choice,
                ax=ax
            )
            st.pyplot(fig)
        
        with viz_col2:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            self.model_evaluator.plot_roc_curve(
                metrics['y_test'],
                metrics['y_pred_proba'],
                model_choice,
                ax=ax
            )
            st.pyplot(fig)
        
        # Model comparison if both models are available
        if len(self.model_evaluator.metrics) > 1:
            st.markdown("---")
            st.subheader("Model Comparison")
            comparison_df = self.model_evaluator.compare_models()
            st.dataframe(comparison_df, use_container_width=True)
    
    def _feature_importance_tab(self, model_choice):
        """Feature importance visualization tab."""
        st.header("Feature Importance")
        
        model_type = 'logistic' if model_choice == "Logistic Regression" else 'random_forest'
        model = self.model_trainer.get_model(model_type)
        
        if model is None:
            st.warning(f"{model_choice} model not trained yet.")
            return
        
        if model_choice == "Random Forest":
            try:
                top_n = st.slider("Number of Top Features", 5, 20, 10)
                fig, ax = plt.subplots(figsize=(10, 8))
                self.model_evaluator.plot_feature_importance(
                    model,
                    model_choice,
                    top_n=top_n,
                    ax=ax
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Feature importance not available: {str(e)}")
        else:
            st.info("Feature importance visualization is available for Random Forest models only.")
            st.markdown("""
            For Logistic Regression, you can interpret coefficients:
            - Positive coefficients increase stroke risk
            - Negative coefficients decrease stroke risk
            - Larger absolute values indicate stronger influence
            """)
    
    def _about_tab(self):
        """About tab with project information."""
        st.header("About This Application")
        
        st.markdown("""
        This application predicts stroke risk using two machine learning models:
        - **Logistic Regression**: Interpretable baseline model
        - **Random Forest**: Advanced model capturing non-linear relationships
        
        **Dataset**: Stroke Prediction Dataset from Kaggle
        
        **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
        
        **Disclaimer**: This application is for educational purposes only. 
        Not a substitute for professional medical advice.
        """)
