from fastapi import APIRouter
from pathlib import Path
import pandas as pd

# Initialize FastAPI router for analytics routes
router = APIRouter()

# Define the file path for the cleaned/processed dataset
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "processed" / "cleaned_highland_crops.csv"

@router.get("/trends")
def get_crop_trends(district: str = None):
    
    # Check if the processed data file exists before reading
    if not PROCESSED_DATA_PATH.exists():
        return {"message": "Processed data not yet available. Please complete preprocessing step."}
    
    # Read the dataset using pandas
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Filter the dataframe by district if a district parameter is provided
    if district:
        df = df[df['District'].str.lower() == district.lower()]
        
    # Group data by Year and Crop, then sum up the production values
    summary = df.groupby(['Year', 'Crop'])['Production'].sum().reset_index()
    
    # Return the district filter status and top 50 records as a dictionary
    return {"district": district, "data": summary.head(50).to_dict(orient="records")}