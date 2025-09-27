# preprocessing.py
"""
Handles the complete data cleaning and feature engineering pipeline.
This script is used by train_model.py (offline) and scoring_logic.py (online).
"""
from typing import Any, Dict, List
import pandas as pd
import re, ast, numpy as np
from datetime import datetime
from config import settings


# --- Configuration for Preprocessing ---
class PreprocessingConfig:
    INITIAL_FEATURE_COLS = [
        "property_id", "zip_code", "neighborhoods",
        "beds", "full_baths", "half_baths", "sqft", "year_built",
        "list_price", "list_date", "sold_price", "estimated_value", "lot_sqft",
        "stories", "hoa_fee", "parking_garage", 'days_on_mls', 'tax',
        # LLM-generated columns are now part of the initial set
        'llm_quality_score', 'llm_risk_score', 'renovation_level',
    ]
    BASE_MODEL_FEATURES = [
        'sqft', 'beds', 'stories', 'estimated_value', 'parking_garage',
        'lot_sqft', 'property_age', 'list_price', 'total_baths',
        'llm_quality_score',    # score from the LLM
        'sqft_x_quality_score' # Interaction between size and LLM's quality 
    ]

    CATEGORICAL_FEATURES = ['zip_code', 'neighborhoods', 'size_range', 'renovation_level']

    TARGET_COLUMN = 'sold_price'
    PRICE_PER_SQFT_OUTLIER_BOUNDS = (20, 350)
    CURRENT_YEAR = datetime.now().year

    


def get_size_range(sqft: float) -> str:
    """Categorizes a property's square footage into a predefined size range."""
    if sqft < 1000: return "lt_1000"
    if 1000 <= sqft < 1500: return "1000-1499"
    if 1500 <= sqft < 2000: return "1500-1999"
    return "2000_plus"



class DataProcessor:
    """A class to handle all data preprocessing for both training and inference."""
    def __init__(self):
        self.config = PreprocessingConfig()
        self.imputation_values: Dict[str, Any] = {}
        self.training_columns: List[str] = []
        self._fitted_categories: Dict[str, List[str]] = {} # To store all possible categories of categorical features
        self.categorical_features_to_encode = self.config.CATEGORICAL_FEATURES.copy()


    def _clean_and_filter(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Handles initial cleaning, deduplication, and outlier removal."""
        df_clean = df.copy()

        # Deduplicate based on most recent listing date
        df_clean['list_date'] = pd.to_datetime(df_clean['list_date'], errors='coerce')
        df_clean.sort_values(by='list_date', ascending=False, inplace=True)
        df_clean.drop_duplicates(subset=['property_id'], keep='first', inplace=True)

        if self.config.TARGET_COLUMN in df_clean.columns:
            df_clean.dropna(subset=[self.config.TARGET_COLUMN, 'sqft'], inplace=True)
            df_clean = df_clean[(df_clean['sqft'] > 100) & (df_clean[self.config.TARGET_COLUMN] > 10000)].copy()
            df_clean['price_per_sqft'] = df_clean[self.config.TARGET_COLUMN] / df_clean['sqft']
            lower, upper = self.config.PRICE_PER_SQFT_OUTLIER_BOUNDS
            df_clean = df_clean[(df_clean['price_per_sqft'] >= lower) & (df_clean['price_per_sqft'] <= upper)]
        return df_clean


    def fit(self, df_raw: pd.DataFrame):
        """
        Learns imputation values using a grouped strategy (by zip and size)
        """
        print("--- Fitting DataProcessor on data with Grouped Imputation ---")
        
        # Start with a clean slate of the necessary columns
        df = df_raw[self.config.INITIAL_FEATURE_COLS].copy()
        
        # Clean the data first to get a reliable base for learning
        df = self._clean_and_filter(df, is_training=True)

        # Engineer the 'size_range' feature which is the primary grouping key
        df['size_range'] = df['sqft'].apply(get_size_range)
        
        # --- Learn Grouped Imputation Values ---
        grouping_keys = ['zip_code', 'size_range']
        
        # Features to impute with the group median
        median_impute_cols = ['lot_sqft', 'year_built', 'list_price', 'estimated_value', 'llm_risk_score', 'tax', 'days_on_mls']
        grouped_medians = df.groupby(grouping_keys)[median_impute_cols].median()

        # Features to impute with the group mode
        mode_impute_cols = ['beds', 'full_baths', 'half_baths', 'stories', 'parking_garage', 'neighborhoods', 'renovation_level']
        grouped_modes = df.groupby(grouping_keys)[mode_impute_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        # --- Learn Global Fallback Imputation Values ---
        global_fallbacks = {
            'lot_sqft': df['lot_sqft'].median(),
            'year_built': df['year_built'].median(),
            'beds': df['beds'].mode()[0],
            'full_baths': df['full_baths'].mode()[0],
            'half_baths': df['half_baths'].mode()[0],
            'stories': df['stories'].mode()[0],
            'parking_garage': df['parking_garage'].mode()[0],
            'neighborhoods': 'Unknown',
            'renovation_level': 'Unknown',
            'list_price': df['list_price'].median(),
            'estimated_value': df['estimated_value'].median(),
            'llm_quality_score': 2, # Conservative default
            'llm_risk_score': df['llm_risk_score'].median(),
            'tax': df['tax'].median(),
            'days_on_mls': df['days_on_mls'].median(),
            'hoa_fee': 0.0,
            'text': ""
        }

        # --- Store all learned values in the instance dictionary ---
        self.imputation_values = {
            "grouped_medians": grouped_medians,
            "grouped_modes": grouped_modes,
            "global_fallbacks": global_fallbacks
        }
        print("Learned grouped and global fallback imputation values.")

        # --- Learn Categorical Encodings ---
        df_filled_for_fitting = df.fillna(self.imputation_values['global_fallbacks'])
        for col in self.config.CATEGORICAL_FEATURES:
            known_categories = df_filled_for_fitting[col].dropna().unique().tolist()
            if 'Unknown' not in known_categories:
                known_categories.append('Unknown')
            self._fitted_categories[col] = known_categories
        print(f"Learned categories for: {list(self._fitted_categories.keys())}")
        
        # --- Fit and Store Training Columns (Schema) ---
        X_schema, _ = self.transform(df) # Use the original cleaned df for the dry run
        self.training_columns = X_schema.columns.tolist()
        print(f"--- DataProcessor fitting complete. Final schema has {len(self.training_columns)} features. ---")

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms raw data into a model-ready feature matrix using grouped imputation."""
        df_transformed = df.copy()
        
        # --- Pre-computation and Initial Cleaning ---
        df_transformed = self._clean_and_filter(df_transformed)
        df_transformed['size_range'] = df_transformed['sqft'].apply(get_size_range)
        
        # --- Intelligent Imputation ---

        # Set index for efficient joining
        df_transformed = df_transformed.set_index(['zip_code', 'size_range'])
        
        # Fill NaNs by joining with the learned grouped stats
        df_transformed.update(self.imputation_values['grouped_medians'], overwrite=False)
        df_transformed.update(self.imputation_values['grouped_modes'], overwrite=False)
        
        # Reset index and apply global fallbacks for anything still missing
        df_transformed.reset_index(inplace=True)
        df_transformed.fillna(self.imputation_values['global_fallbacks'], inplace=True)

        # --- Feature Engineering ---
        df_transformed['property_age'] = self.config.CURRENT_YEAR - df_transformed['year_built']
        df_transformed['total_baths'] = df_transformed['full_baths'] + (0.5 * df_transformed['half_baths'])
        df_transformed['sqft_x_quality_score'] = df_transformed['sqft'] * df_transformed['llm_quality_score']

        # --- One-Hot Encoding and Final Alignment ---
        for col, categories in self._fitted_categories.items():
            # Handle unseen categories before creating the categorical type
            df_transformed[col] = df_transformed[col].apply(lambda x: x if x in categories else 'Unknown')
            df_transformed[col] = pd.Categorical(df_transformed[col], categories=categories)
            
        df_transformed = pd.get_dummies(df_transformed, columns=self.config.CATEGORICAL_FEATURES, prefix_sep='_', dtype=int)


        # Align to the final training schema
        if self.training_columns:
            X = df_transformed.reindex(columns=self.training_columns, fill_value=0)
        else: # Dry run path
            base_features = self.config.BASE_MODEL_FEATURES
            ohe_features = [col for col in df_transformed.columns if any(cat in col for cat in self.config.CATEGORICAL_FEATURES)]
            final_feature_list = base_features + ohe_features
            X = df_transformed[[col for col in final_feature_list if col in df_transformed.columns]]

        y = df_transformed[self.config.TARGET_COLUMN] if self.config.TARGET_COLUMN in df_transformed.columns else None
            
        return X, y
    
    
    def prepare_inference_data(self, df_raw: pd.DataFrame) -> List[dict]:
        """
        Public method that takes a raw for-sale DataFrame, cleans it,
        and returns a list of Pydantic-ready dictionaries.
        """
        df = df_raw.copy()
        
        # Deduplicate
        df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
        df.sort_values(by='list_date', ascending=False, inplace=True)
        df.drop_duplicates(subset=['property_id'], keep='first', inplace=True)
        
        # Impute with LEARNED values from the fitted processor
        df['estimated_value'] = df['estimated_value'].fillna(df['list_price'])
        df.fillna(self.imputation_values, inplace=True)

        list_columns = ['positive_features', 'negative_features']
        
        for col in list_columns:
            if col in df.columns:
                # The .apply() method will run this safe_literal_eval function on every row of the column.
                def safe_literal_eval(val):
                    try:
                        # If it's already a list (rare, but possible), just return it.
                        if isinstance(val, list):
                            return val
                        # ast.literal_eval will safely parse the string into a list.
                        return ast.literal_eval(str(val))
                    except (ValueError, SyntaxError):
                        # If the string is malformed (e.g., empty or just text), return an empty list.
                        return []
                
                df[col] = df[col].apply(safe_literal_eval)

        # --- Final Type Conversion and Formatting ---
        # convert the DataFrame to a list of dictionaries and handle any final formatting.
        # Replace any remaining Pandas-specific nulls (like NaT) with Python's None
        df = df.replace({np.nan: None, pd.NaT: None})
        
        dict_records = df.to_dict(orient='records')
        
        # The final loop is now much simpler, mainly handling integer casting.
        clean_records = []
        for record in dict_records:
            if record.get('list_date'):
                record['list_date'] = record['list_date'].strftime('%Y-%m-%d')

            int_keys = ['beds', 'full_baths', 'half_baths', 'stories', 'parking_garage', 
                        'days_on_mls', 'property_id', 'zip_code', 'year_built']
            for key in int_keys:
                if key in record and record[key] is not None:
                    try:
                        record[key] = int(record[key])
                    except (ValueError, TypeError):
                        record[key] = 0 # Fallback for safety
            
            clean_records.append(record)
            
        print(f"Prepared {len(clean_records)} clean property records for inference.")
        return clean_records