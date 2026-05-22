# Car Price Prediction with Classical Machine Learning

Predicting used-car prices from the **London Cars 2014** dataset (9,080 listings) using classical machine learning — both as a regression problem (predicting the exact price) and a classification problem (predicting a price band). The project covers the full pipeline: exploratory analysis, preprocessing, model training, hyperparameter tuning, and feature selection.

## Overview

The dataset has 9,080 rows and 11 columns (Make, Model, Year, Mileage, Body Style, Engine, Transmission, colours, etc.). The goal is to model how these attributes drive a car's price.

The work is split into two framings of the same problem:

- **Regression** — predict the numerical price directly.
- **Classification** — bucket prices into low / medium / high bands and predict the band.

## What's inside

**Exploratory data analysis**
Distribution checks, missing-value inspection, and feature relationships to understand the data before modelling.

**Preprocessing**
- Label encoding for high-cardinality categoricals (`Model`) and one-hot encoding for the rest, with `drop_first=True` to avoid the dummy-variable trap.
- Feature standardisation with `StandardScaler` (fit on training data only to prevent leakage).
- A 70/30 train/test split, stratified for the classification task to preserve class balance.

**Regression models**
- Linear Regression as a baseline.
- Ridge Regression with the regularisation strength tuned via cross-validation.
- Random Forest Regressor with hyperparameters selected through `GridSearchCV`.

**Classification models**
- Logistic Regression (L1-regularised, tuned `C`).
- Random Forest Classifier with tuned hyperparameters.
- Evaluation with accuracy, precision, recall, F1, and confusion matrices.

**Feature selection**
Identifying which attributes contribute most to predicting price.

## Key findings

- Year, Mileage, Model, and Engine are the strongest predictors of price.
- Tree-based models outperformed linear models on the regression task, though they showed a tendency to overfit (large train/test gap).
- The price classes are heavily imbalanced — the "high" price band is a small minority — which makes the minority class the hardest to predict, and makes macro-averaged metrics more informative than raw accuracy.
- Cross-validation and regularisation were used throughout to give a fair, leakage-free estimate of generalisation performance.

## Tech stack

- Python, pandas, NumPy
- scikit-learn (`LinearRegression`, `Ridge`, `RandomForest`, `LogisticRegression`, `GridSearchCV`, `KFold`, `StandardScaler`)
- matplotlib / seaborn for visualisation
- Jupyter Notebook

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Open the notebook and run the cells in order.
