"""Data preprocessing and regex string cleaning module."""

import pandas as pd

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset: fix string numbers, remove aggregate rows."""
    df = df.copy()
    
    # 1. Remove aggregate / island total rows if present
    if 'District' in df.columns:
        df = df[~df['District'].astype(str).str.contains("Total|Island", case=False, na=False)]
        
    # 2. Convert comma-separated string numbers in numeric columns to float
    numeric_cols = ['Extent', 'Production']
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop rows where critical values are null
    df = df.dropna(subset=['Extent', 'Production'])
    return df

if __name__ == "__main__":
    print("Preprocessing module ready.")


def temporal_train_test_split(df: pd.DataFrame):
    """Split dataset temporally based on Year to prevent data leakage."""
    train = df[df['Year'] <= 2017]
    val = df[(df['Year'] >= 2018) & (df['Year'] <= 2020)]
    test = df[df['Year'] >= 2021]
    
    print(f"Train set: {train.shape}, Validation set: {val.shape}, Test set: {test.shape}")
    return train, val, test