Stroke Prediction Machine Learning Application

A machine learning application that predicts stroke risk using Logistic Regression and Random Forest models trained on the Stroke Prediction Dataset from Kaggle.


Installation

1. Clone the repository (or navigate to project directory)

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download the Stroke Prediction Dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
   - Place `healthcare-dataset-stroke-data.csv` in the `data/` directory
   - Rename it to `stroke.csv`

Dataset

The dataset contains the following features:

- Age: Patient age
- Gender: Male/Female/Other
- Hypertension: 0/1
- Heart Disease: 0/1
- Ever Married: Yes/No
- Work Type: Private/Self-employed/Govt_job/children/Never_worked
- Residence Type: Urban/Rural
- Avg Glucose Level: Average glucose level
- BMI: Body Mass Index
- Smoking Status: formerly smoked/never smoked/smokes/Unknown
- Stroke: Target variable (0/1)

Usage

1.  Train Models

Run the main training script:

```bash
python src/main.py
```

This will:

- Load and preprocess the data
- Train both Logistic Regression and Random Forest models
- Perform hyperparameter tuning with GridSearchCV
- Evaluate models with comprehensive metrics
- Save trained models to `models/` directory
- Generate visualizations in `results/` directory

2.  Run Streamlit Application

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

Note: Make sure you've trained the models first (step 1) for full functionality.

The app provides:

- Prediction Tab: Input patient data and get stroke risk predictions
- Performance Tab: View model metrics, confusion matrices, and ROC curves
- Feature Importance Tab: Visualize feature importance (Random Forest)
- About Tab: Project information and disclaimers

Model Evaluation

Models are evaluated using:

- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the ROC curve
- Confusion Matrix: Detailed classification breakdown

Key Features

- Two ML Models: Logistic Regression and Random Forest with hyperparameter tuning
- Comprehensive Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- Model Comparison: Side-by-side comparison of both models
- Interactive App: Streamlit web interface with real-time predictions
- Feature Importance: Visualization of important features (Random Forest)

Code Architecture

The project uses object-oriented design with four main classes:

- DataProcessor: Handles data loading, cleaning, encoding, and scaling
- ModelTrainer: Trains and tunes Logistic Regression and Random Forest models
- ModelEvaluator: Computes metrics and creates visualizations
- StreamlitApp: Interactive web interface for predictions

References

- [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

Disclaimer

This application is for educational purposes only.

It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

