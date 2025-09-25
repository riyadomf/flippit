# V2_preprocessing.py
"""
Handles the complete data cleaning and feature engineering pipeline.
This script is used by train_model.py (offline) and scoring_logic.py (online).
"""
import pandas as pd
import numpy as np


# This CONFIG class is specific to the preprocessing steps.
class PreprocessingConfig:
    INITIAL_FEATURE_COLS = [
        "property_id", "text", "zip_code", "neighborhoods",
        "beds", "full_baths", "half_baths", "sqft", "year_built",
        "list_price", "sold_price", "estimated_value", "tax", "lot_sqft",
        "stories", "hoa_fee", "parking_garage"
    ]
    TARGET_COLUMN = 'sold_price'
    PRICE_PER_SQFT_OUTLIER_BOUNDS = (75, 350)
    CURRENT_YEAR = 2025
    POSITIVE_KEYWORD_MAP = {
        'has_granite': 'granite',
        'has_hardwood': 'hardwood|wood floor',
        'has_stainless': 'stainless',
    }

def get_size_range(sqft: float) -> str:
    """Categorizes a property's square footage into a predefined size range."""
    if sqft < 1000: return "<1000"
    if 1000 <= sqft < 1500: return "1000-1499"
    if 1500 <= sqft < 2000: return "1500-1999"
    return "2000+"


def preprocess_for_resale_model(df_raw: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Executes the full preprocessing pipeline on the raw sold_properties DataFrame.
    Returns a model-ready feature matrix (X) and target vector (y).
    """
    config = PreprocessingConfig()
    df = df_raw[config.INITIAL_FEATURE_COLS].copy()

    # 1. CLEANING & OUTLIER REMOVAL
    df.dropna(subset=[config.TARGET_COLUMN, 'sqft'], inplace=True)
    df = df[(df['sqft'] > 100) & (df[config.TARGET_COLUMN] > 10000)].copy()
    df['price_per_sqft'] = df[config.TARGET_COLUMN] / df['sqft']
    lower, upper = config.PRICE_PER_SQFT_OUTLIER_BOUNDS
    df = df[(df['price_per_sqft'] >= lower) & (df['price_per_sqft'] <= upper)]

    # 2. IMPUTATION
    imputation_values = {
        'beds': df['beds'].median(), 'full_baths': df['full_baths'].median(),
        'half_baths': 0, 'lot_sqft': df['lot_sqft'].median(),
        'year_built': df['year_built'].median(), 'hoa_fee': 0.0,
        'stories': df['stories'].mode()[0], 'parking_garage': df['parking_garage'].mode()[0],
        'neighborhoods': df['neighborhoods'].mode()[0], 'text': ""
    }
    df.fillna(imputation_values, inplace=True)
    df['estimated_value'].fillna(df['list_price'], inplace=True)
    df['estimated_value'].fillna(df[config.TARGET_COLUMN].median(), inplace=True)

    # 3. FEATURE ENGINEERING
    df['property_age'] = config.CURRENT_YEAR - df['year_built']
    df['total_baths'] = df['full_baths'] + (0.5 * df['half_baths'])
    df['size_range'] = df['sqft'].apply(get_size_range)
    for col, term in config.POSITIVE_KEYWORD_MAP.items():
        df[col] = df['text'].str.contains(term, case=False, regex=True).astype(int)

    # 4. ONE-HOT ENCODING
    categorical_features = ['zip_code', 'neighborhoods', 'size_range']
    df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features, dtype=int)

    # 5. FINAL ASSEMBLY
    y = df[config.TARGET_COLUMN]
    cols_to_drop = [
        config.TARGET_COLUMN, 'property_id', 'text', 'list_price', 'price_per_sqft',
        'year_built', 'full_baths', 'half_baths', 'list_date'
    ]
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    
    return X, y
