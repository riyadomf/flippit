# V2_preprocessing.py
"""
Handles the complete data cleaning and feature engineering pipeline.
This script is used by train_model.py (offline) and scoring_logic.py (online).
"""
from typing import List
import pandas as pd
from datetime import datetime
from config import settings


# --- Configuration for Preprocessing ---
class PreprocessingConfig:
    INITIAL_FEATURE_COLS = [
        "property_id", "text", "zip_code", "neighborhoods",
        "beds", "full_baths", "half_baths", "sqft", "year_built",
        "list_price", "list_date", "sold_price", "estimated_value", "tax", "lot_sqft",
        "stories", "hoa_fee", "parking_garage"
    ]
    BASE_FEATURES = [
        'sqft', 'beds', 'stories', 'hoa_fee', 'estimated_value', 'parking_garage',
        'property_age', 'total_baths',
        # And now we add the condition flags as features
        'is_fixer_upper', 'is_renovated'
    ]

    TARGET_COLUMN = 'sold_price'
    PRICE_PER_SQFT_OUTLIER_BOUNDS = (75, 350)
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
        self.high_cost_search_term = '|'.join(settings.HIGH_COST_KEYWORDS)
        self.low_cost_search_term = '|'.join(settings.LOW_COST_KEYWORDS)
        self.imputation_values = {}
        self.training_columns = []
        self._fitted_categories = {} # To store all possible categories


    def _clean_and_filter(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Handles initial cleaning, deduplication, and (optional) outlier removal."""
        df_clean = df.copy()
        df_clean['list_date'] = pd.to_datetime(df_clean['list_date'], errors='coerce')
        df_clean.sort_values(by='list_date', ascending=False, inplace=True)
        df_clean.drop_duplicates(subset=['property_id'], keep='first', inplace=True)

        if is_training:
            df_clean.dropna(subset=[self.config.TARGET_COLUMN, 'sqft'], inplace=True)
            df_clean = df_clean[(df_clean['sqft'] > 100) & (df_clean[self.config.TARGET_COLUMN] > 10000)].copy()
            df_clean['price_per_sqft'] = df_clean[self.config.TARGET_COLUMN] / df_clean['sqft']
            lower, upper = self.config.PRICE_PER_SQFT_OUTLIER_BOUNDS
            df_clean = df_clean[(df_clean['price_per_sqft'] >= lower) & (df_clean['price_per_sqft'] <= upper)]
        return df_clean
    

    def fit(self, df_raw: pd.DataFrame):
        """Learns all necessary statistics and schemas from the full training dataset."""
        print("--- Fitting DataProcessor on training data ---")
        df = df_raw[self.config.INITIAL_FEATURE_COLS].copy()
        
        # 2. Clean, filter to "gold standard" for learning imputation values
        df_clean = self._clean_and_filter(df, is_training=True)

        # 3. Learn imputation values from the clean subset
        self.imputation_values = {
            'beds': df_clean['beds'].median(), 'full_baths': df_clean['full_baths'].median(),
            'half_baths': 0, 'lot_sqft': df_clean['lot_sqft'].median(),
            'year_built': df_clean['year_built'].median(), 'hoa_fee': 0.0,
            'stories': df_clean['stories'].mode()[0], 'parking_garage': df_clean['parking_garage'].mode()[0],
            'neighborhoods': 'Unknown', 'text': "", 'list_price': df_clean['list_price'].median(),
            'estimated_value': df_clean['estimated_value'].median()
        }
        print("Imputation values learned.")

        # # 2. IMPUTATION
        # print("Imputing missing values with granular medians/modes...")
        # # Fill based on ZIP code median for key housing stats
        # for col in ['beds', 'full_baths', 'lot_sqft', 'year_built']:
        #     df[col] = df.groupby('zip_code')[col].transform(lambda x: x.fillna(x.median()))
        
        # # Fallback for any ZIP codes that had no data
        # df.fillna({'beds': df['beds'].median(), 'full_baths': df['full_baths'].median(),
        #             'lot_sqft': df['lot_sqft'].median(), 'year_built': df['year_built'].median(),
        #             'stories': df['stories'].mode()[0], 'parking_garage': df['parking_garage'].mode()[0]},
        #             inplace=True)
        
        # imputation_values = {
        #     'half_baths': 0, 'hoa_fee': 0.0, 'neighborhoods': 'Unknown',
        #     'text': ""
        # }
        # df.fillna(imputation_values, inplace=True)

        # df['estimated_value'].fillna(df['list_price'], inplace=True)
        # # Now, fill remaining NaNs with the median sold_price of their ZIP code
        # df['estimated_value'] = df.groupby('zip_code')['estimated_value'].transform(lambda x: x.fillna(x.median()))
        # # Final global fallback for any ZIP that had no data at all
        # df['estimated_value'].fillna(df[config.TARGET_COLUMN].median(), inplace=True)

        # df['list_price'].fillna(df['estimated_value'], inplace=True)
        # df['list_price'].fillna(df[config.TARGET_COLUMN].median(), inplace=True)

        # print("Missing values imputed successfully.")
        

        # Fit and Store Training Columns
        # Perform a dry run on the clean (but unfiltered) data to get all possible columns
        X_schema, _ = self.transform(df_clean)
        self.training_columns = X_schema.columns.tolist()
        print(f"--- DataProcessor fitting complete. Final schema has {len(self.training_columns)} features. ---")


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms raw data into a model-ready feature matrix."""
        df_transformed = df.copy()

        # Impute missing values using LEARNED values
        # Special fallback for estimated_value to use list_price first
        df_transformed['estimated_value'].fillna(df_transformed['list_price'], inplace=True)
        
        # Apply all other learned imputation values
        df_transformed.fillna(self.imputation_values, inplace=True)

        # Engineer Features
        df_transformed['property_age'] = self.config.CURRENT_YEAR - df_transformed['year_built']
        df_transformed['total_baths'] = df_transformed['full_baths'] + (0.5 * df_transformed['half_baths'])
        df_transformed['size_range'] = df_transformed['sqft'].apply(get_size_range)
        for col, term in settings.KEYWORD_MAP.items():
            df_transformed[col] = df_transformed['text'].str.contains(term, case=False, regex=True).astype(int)

        # 3. One-Hot Encode using the LEARNED categories to ensure consistency
        for col, categories in self._fitted_categories.items():
            df_transformed[col] = pd.Categorical(df_transformed[col], categories=categories)
        df_transformed = pd.get_dummies(df_transformed, columns=self._fitted_categories.keys(), prefix=self._fitted_categories.keys(), dtype=int)
        
        # Final Feature Selection and Alignment
        if self.training_columns:
            X = df_transformed.reindex(columns=self.training_columns, fill_value=0)
        else: # This path is only for the schema-defining dry run inside .fit()
            base_features = self.config.BASE_FEATURES.copy()
            premium_keyword_features = list(settings.PREMIUM_FINISH_KEYWORDS.keys())
            ohe_features = [col for col in df_transformed.columns if any(cat in col for cat in self._fitted_categories.keys())]
            
            final_feature_list = base_features + premium_keyword_features + ohe_features
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
        df['estimated_value'].fillna(df['list_price'], inplace=True)
        df.fillna(self.imputation_values, inplace=True)

        # Explicitly handle data types to create Pydantic-safe dictionaries
        # This is the manual cleaning logic, now moved to its rightful home.
        dict_records = df.to_dict(orient='records')
        clean_records = []
        for record in dict_records:
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean_record[key] = None
                else:
                    clean_record[key] = value
            
            if clean_record.get('list_date'):
                clean_record['list_date'] = clean_record['list_date'].strftime('%Y-%m-%d')

            for key in ['beds', 'full_baths', 'half_baths', 'stories', 'parking_garage', 'days_on_mls', 'property_id', 'zip_code', 'year_built']:
                 if key in clean_record:
                    clean_record[key] = int(clean_record.get(key) or 0)
            
            clean_records.append(clean_record)
            
        return clean_records
