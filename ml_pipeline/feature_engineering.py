"""Feature engineering module for yield ratios and lag features."""

import pandas as pd
import numpy as np


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


def apply_target_encoding_and_log(df: pd.DataFrame, target_col='Production'):
    """Apply log transformation to target and encode categorical features."""
    df = df.copy()
    
    # 1. Technique 6: Log Transformation for target skewness normalization
    if target_col in df.columns:
        df['Log_Production'] = np.log1p(df[target_col])
        
    # 2. Technique 4: Smoothed Target Encoding for District and Crop
    for col in ['District', 'Crop']:
        if col in df.columns and target_col in df.columns:
            mean_encoding = df.groupby(col)[target_col].mean()
            df[f'{col}_Target_Encoded'] = df[col].map(mean_encoding)
            
    return df

