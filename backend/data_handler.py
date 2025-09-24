# data_handler.py

import pandas as pd
from config import settings
from scoring_logic import get_size_range

# This dictionary will hold pre-calculated market data in memory.
market_data_store = {}

def load_market_data(filepath: str = settings.SOLD_PROPERTIES_CSV):
    """
    Loads and processes the sold properties data to establish a market baseline.
    """
    try:
        sold_df = pd.read_csv(filepath)
        
        # --- Data Cleaning (Based on EDA findings) ---
        sold_df.dropna(subset=['price_per_sqft'], inplace=True)
        sold_df = sold_df[(sold_df['price_per_sqft'] > 50) & (sold_df['price_per_sqft'] < 500)] # Remove unrealistic low values

        # Feature Engineering 
        # Assign each sold property to a size range using our helper function
        sold_df['size_range'] = sold_df['sqft'].apply(get_size_range)

        # --- Calculations for Model ---
        # 1. Primary Metric: Group by ZIP and size_range
        median_by_zip_and_size = sold_df.groupby(['zip_code', 'size_range'])['price_per_sqft'].median()

        # 2. Fallback Metric: Group by just ZIP
        median_by_zip = sold_df.groupby('zip_code')['price_per_sqft'].median()
        
        # 3. Global Fallback Metric: Overall median
        overall_median = sold_df['price_per_sqft'].median()

        # Store the processed data in dictionary for the app to use
        nested_market_data = {}
        for (zip_code, size_range), median_val in median_by_zip_and_size.items():
            if zip_code not in nested_market_data:
                nested_market_data[zip_code] = {}
            nested_market_data[zip_code][size_range] = median_val
            
        market_data_store['median_by_zip_and_size'] = nested_market_data
        market_data_store['median_by_zip'] = median_by_zip.to_dict()
        market_data_store['overall_median'] = overall_median
        
        print("Market data loaded and processed successfully.")
        
    except FileNotFoundError:
        print(f"ERROR: The file {filepath} was not found. The application cannot start without it.")
        raise