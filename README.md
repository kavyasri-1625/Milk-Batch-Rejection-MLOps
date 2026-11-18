
# Milk Batch Rejection Prediction - MLOps

## Project Objective
Predict whether a milk batch is likely to be accepted or rejected
during quality inspection at a village dairy collection centre.

## Dataset
- Version 1: Normal conditions (250 records)
- Version 2: Summer conditions (250 records)
- Combined dataset: 500 records

## Features
- Milk quantity
- Collection temperature
- Transportation duration
- Cooling delay
- Fat percentage
- Acidity indicator
- Container type
- Ambient temperature

## Feature Engineering
- Temperature Exposure Index
- Transport Delay Risk

## Machine Learning Models
1. Logistic Regression
2. Random Forest

## MLOps Tools
- Python
- Pandas
- Scikit-learn
- MLflow
- FastAPI
- GitHub Actions

## API
POST /predict

Output:
- Likely Accepted
- Rejection Risk

## Testing
Run:
pytest -v test_app.py

## Run API
uvicorn app:app --reload
