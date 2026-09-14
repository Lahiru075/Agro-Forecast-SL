"""Model training and multi-model benchmarking module."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def benchmark_models(X_train, y_train, X_val, y_val):
    """Benchmark baseline and ensemble regression models using TimeSeriesSplit."""
    models = {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        results[name] = rmse
        print(f"Model: {name} | Validation RMSE: {rmse:.4f}")
        
    return results

