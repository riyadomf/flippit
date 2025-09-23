# main.py

from fastapi import FastAPI, HTTPException, Query
from typing import List
import pandas as pd
import numpy as np
import re

# Import custom modules
from schemas import PropertyDataInput, ScoreOutput
from data_handler import load_market_data, market_data_store
from scoring_logic import score_property

# --- Application Setup ---
app = FastAPI(
    title="Flippit Real Estate Deal Scorer",
    description="An API to score for-sale properties on their flip potential.",
    version="1.0.0"
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    On application startup, load the market data from sold_properties.csv.
    This ensures the data is loaded once into memory, not on every API call.
    """
    load_market_data()

# --- Helper function for cleaning the for-sale data ---
def clean_for_sale_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean['hoa_fee'].fillna(0, inplace=True)
    
    def extract_tax(tax_string):
        if isinstance(tax_string, str):
            numbers = re.findall(r'\d+\.?\d*', tax_string)
            return float(numbers[0]) if numbers else 0
        return tax_string if not pd.isna(tax_string) else 0

    df_clean['tax'] = df_clean['tax'].apply(extract_tax)
    # Fill any other critical NaNs with safe defaults
    df_clean.fillna({
        'estimated_value': 0, 'days_on_mls': 0, 'text': '', 'half_baths': 0
    }, inplace=True)
    return df_clean

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/score", response_model=ScoreOutput)
async def score_single_property(property_input: PropertyDataInput):
    """
    Scores a single property based on the provided JSON data.
    """
    if not market_data_store:
        raise HTTPException(status_code=500, detail="Market data is not loaded.")
    
    return score_property(property_input, market_data_store)


@app.get("/properties", response_model=List[ScoreOutput])
async def get_scored_properties(
    min_roi: float = Query(None, description="Minimum ROI percentage to filter by"),
    max_price: float = Query(None, description="Maximum list price to filter by"),
    min_beds: int = Query(None, description="Minimum number of bedrooms")
):
    """
    Gets all for-sale properties, scores them, 
    and returns a list that can be filtered.
    """
    if not market_data_store:
        raise HTTPException(status_code=500, detail="Market data is not loaded.")

    try:
        forsale_df = pd.read_csv("dataset/for_sale_properties.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="for_sale_properties.csv not found.")
    
    # Clean the dataframe
    forsale_df_clean = clean_for_sale_data(forsale_df)
    
    scored_properties = []
    for _, row in forsale_df_clean.iterrows():
        # Convert row to a Pydantic model for validation and scoring
        property_input = PropertyDataInput(**row.to_dict())
        scored_result = score_property(property_input, market_data_store)
        scored_properties.append(scored_result)
        
    # --- Filtering Logic ---
    filtered_results = scored_properties
    if min_roi is not None:
        filtered_results = [p for p in filtered_results if p.roi_percentage >= min_roi]
    if max_price is not None:
        filtered_results = [p for p in filtered_results if p.list_price <= max_price]
    if min_beds is not None:
        filtered_results = [p for p in filtered_results if row['beds'] >= min_beds] # Requires beds in schema
        
    return filtered_results