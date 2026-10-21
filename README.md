
# Student Performance Prediction

## Overview
A machine learning project that predicts a student's final mathematics grade
using academic, demographic, and social information.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit

## Model Performance
- MAE: approximately 1.21
- RMSE: approximately 2.03
- R²: approximately 0.80

## How to Run Locally
1. Install the dependencies:
   `pip install -r requirements.txt`
2. Run the application:
   `streamlit run app.py`

## Limitations
Predictions are estimates, not guaranteed grades. The model was trained
on a dataset of students from two Portuguese schools, so performance
may differ for other student populations.
