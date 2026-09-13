"""Feature engineering module for yield ratios and lag features."""

import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features like yield ratio and temporal lags."""
    df = df.copy()
    
    # 1. Technique 1: Domain Yield Ratio (Production per Hectare)
    if 'Production' in df.columns and 'Extent' in df.columns:
        df['Crop_Yield'] = df['Production'] / (df['Extent'] + 1e-5) # Prevent division by zero
        
    # 2. Technique 2: Temporal Lag Features (1-year lag grouped by District and Crop)
    if all(col in df.columns for col in ['District', 'Crop', 'Year', 'Production']):
        df = df.sort_values(by=['District', 'Crop', 'Year'])
        df['Production_Lag_1Year'] = df.groupby(['District', 'Crop'])['Production'].shift(1)
        
    # Fill missing lag values with median or 0
    df['Production_Lag_1Year'] = df['Production_Lag_1Year'].fillna(df['Production'].median())
    
    return df



def temporal_train_test_split(df: pd.DataFrame):
    """Split dataset temporally based on Year to prevent data leakage."""
    train = df[df['Year'] <= 2017]
    val = df[(df['Year'] >= 2018) & (df['Year'] <= 2020)]
    test = df[df['Year'] >= 2021]
    
    print(f"Train set: {train.shape}, Validation set: {val.shape}, Test set: {test.shape}")
    return train, val, test