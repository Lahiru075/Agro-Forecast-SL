from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "researchData.xlsx"

def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
   
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {file_path}")
    
    df = pd.read_excel(file_path)
    print(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())