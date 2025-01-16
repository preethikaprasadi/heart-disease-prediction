# Heart Disease Prediction and Model Evaluation
A machine learning project for predicting heart disease using Python.

## Overview
This project focuses on predicting the likelihood of heart disease using a dataset containing various clinical and demographic features. It implements data preprocessing, model training, evaluation, and hyperparameter tuning using Python and several machine learning libraries.

## Features
- Data loading and exploration
- Preprocessing (handling categorical and numeric features)
- Model training using Logistic Regression
- Model evaluation with metrics such as accuracy, confusion matrix, ROC curve, and AUC score
- Hyperparameter tuning for Random Forest Classifier
- Saving the trained model for future use

## Requirements
To run this project, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install the libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Dataset
The dataset used for this project is stored in `heart.csv`. It includes the following key features:
- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male, 0 = female)
- **cp**: Chest pain type (categorical)
- **trestbps**: Resting blood pressure (numeric)
- **chol**: Serum cholesterol (numeric)
- **thalach**: Maximum heart rate achieved (numeric)
- **oldpeak**: ST depression induced by exercise relative to rest (numeric)
- **restecg**: Resting electrocardiographic results (categorical)
- **thal**: Thalassemia (categorical)
- **target**: Diagnosis of heart disease (1 = present, 0 = not present)

## Usage
### 1. Load and Explore Data
The script begins by loading the dataset, exploring its structure, and checking for missing values.

### 2. Preprocessing
- **Categorical features** are one-hot encoded.
- **Numeric features** are standardized using `StandardScaler`.

### 3. Model Training
- A `Pipeline` is created to combine preprocessing steps and the `LogisticRegression` model.
- The model is trained using an 80-20 train-test split.

### 4. Model Evaluation
- **Metrics:**
  - Accuracy
  - Confusion Matrix (visualized using seaborn heatmap and `ConfusionMatrixDisplay`)
  - ROC Curve and AUC Score
  - Precision-Recall Curve

### 5. Hyperparameter Tuning
- A `RandomizedSearchCV` is used to optimize hyperparameters for a `RandomForestClassifier`.
- The best model is evaluated and saved to a file (`heart_disease_model.pkl`) using `joblib`.

## Running the Script
Run the Python script to execute all steps:
```bash
python heart-disease-prediction.ipynb
```

## Key Outputs
- **Model Metrics:**
  - Accuracy of Logistic Regression and Random Forest models
  - Confusion Matrix
  - ROC Curve and AUC Score
  - Precision-Recall Curve
- **Best Hyperparameters:** Found through RandomizedSearchCV for the Random Forest model
- **Saved Model File:** `heart_disease_model.pkl`

## Visualization
The script includes visualizations for:
1. Confusion Matrix
2. ROC Curve
3. Precision-Recall Curve

## File Descriptions
- **heart.csv**: Dataset file
- **heart-disease-prediction.ipynb**: Main script for model training and evaluation
- **heart_disease_model.pkl**: Saved Random Forest model after hyperparameter tuning

## Future Improvements
- Use cross-validation for more robust model evaluation.
- Add more advanced models (e.g., Gradient Boosting, Neural Networks).
- Explore feature importance for model interpretability.

## License
This project is open-source and available for educational purposes.
