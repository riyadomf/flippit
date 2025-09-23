# data_handler.py

import pandas as pd
from config import settings

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
        sold_df = sold_df[sold_df['price_per_sqft'] > 100]  # Remove unrealistic low values

        # --- Calculations for Model ---
        median_price_per_sqft_by_zip = sold_df.groupby('zip_code')['price_per_sqft'].median()
        overall_median_price_per_sqft = sold_df['price_per_sqft'].median()

        # Store the processed data in dictionary for the app to use
        market_data_store['median_by_zip'] = median_price_per_sqft_by_zip.to_dict()
        market_data_store['overall_median'] = overall_median_price_per_sqft
        
        print("Market data loaded and processed successfully.")
        
    except FileNotFoundError:
        print(f"ERROR: The file {filepath} was not found. The application cannot start without it.")
        raise