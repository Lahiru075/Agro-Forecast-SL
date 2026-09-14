"""Outlier handling and SHAP diagnostic analysis module."""

import pandas as pd
import shap
import xgboost as xgb

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out extreme production and extent anomalies."""
    df = df.copy()
    # Remove rows where extent or production are unrealistically zero/negative
    df = df[(df['Extent'] > 0) & (df['Production'] >= 0)]
    return df

def compute_shap_values(model, X_test):
    """Compute SHAP values for model interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values
