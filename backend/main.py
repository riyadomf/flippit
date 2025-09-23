# main.py
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import re, time
from config import settings

# Import all our custom modules
import models
import schemas
import scoring_logic
import data_handler
from database import engine, get_db

# This line creates the database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# --- In-Memory Task Status Tracker ---
task_status = {"is_scoring": False}

app = FastAPI(
    title="Flippit Real Estate Deal Scorer",
    description="A production-ready API to score and serve real estate deals.",
    version="1.1.0"
)

origins = [
    "http://localhost:3000", # The origin of your React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Allows specific origins
    allow_credentials=True, # Allows cookies/authentication headers
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


@app.on_event("startup")
async def startup_event():
    """Load market data into memory once when the application starts."""
    data_handler.load_market_data()

# --- Helper function for cleaning the for-sale data ---
# In main.py, update the clean_for_sale_data function

import pandas as pd
import numpy as np
import re
import schemas # Make sure to import your schemas file

def clean_for_sale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the for-sale properties DataFrame.
    - Handles missing values & corrects data types.
    - Filters for only 'FOR_SALE' properties.
    - Removes duplicate property_ids, keeping the most recent listing.
    """
    # My Thought Process: Start by defining all columns we will ever need.
    # This includes columns for filtering/sorting (like 'status') and all fields
    # required by our Pydantic input schema.
    required_schema_columns = list(schemas.PropertyDataInput.model_fields.keys())
    processing_columns = ['status']
    all_needed_columns = list(set(required_schema_columns + processing_columns))
    
    # Ensure all required columns exist in the DataFrame to prevent KeyErrors.
    for col in all_needed_columns:
        if col not in df.columns:
            df[col] = np.nan # Add missing columns and fill with NaN for now
            
    df_clean = df[all_needed_columns].copy()

    # --- 1. Filter for Actionable Properties ---
    # My Thought Process: Do the broad filtering first to reduce the dataset size.
    df_clean = df_clean[df_clean['status'] == 'FOR_SALE'].copy()

    # --- 2. Correct Data Types Robustly ---
    # My Thought Process: Forcibly convert columns to their correct types.
    # Using `errors='coerce'` is critical. It will turn any value that *cannot* be
    # converted into a number (e.g., a string like 'N/A') into NaN. This prevents
    # the app from crashing and allows us to handle all missing data in one step later.
    numeric_cols = ['list_price', 'sqft', 'zip_code', 'year_built', 'hoa_fee', 'tax', 
                    'estimated_value', 'days_on_mls', 'beds', 'full_baths', 
                    'latitude', 'longitude']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    df_clean['list_date'] = pd.to_datetime(df_clean['list_date'], errors='coerce')

    # --- 3. Handle Duplicates Logically ---
    # My Thought Process: The user's original logic was good. We just avoid `inplace=True`.
    # This ensures the most recent listing for any given property_id is kept.
    df_clean = df_clean.sort_values(by='list_date', ascending=False)
    df_clean = df_clean.drop_duplicates(subset=['property_id'], keep='first')
    
    # --- 4. The Master FillNA Step ---
    # My Thought Process: This is the most important change. We create a single,
    # comprehensive dictionary to fill ALL potential NaN values with a safe,
    # type-appropriate default. This single step fixes the original error and
    # makes the function much more resilient to bad data.
    fill_values = {
        'list_price': 0.0,
        'sqft': 0.0,
        'zip_code': 0,
        'year_built': 0,
        'hoa_fee': 0.0,
        'tax': 0.0,
        'estimated_value': 0.0,
        'days_on_mls': 0,
        'beds': 0,
        'full_baths': 0,
        'latitude': 0.0,
        'longitude': 0.0,
        'text': "",
        'full_street_line': "",
        'city': "",
        'state': "",
        'list_date': pd.NaT # Fill missing dates with NaT
    }
    df_clean.fillna(value=fill_values, inplace=True)
    
    # --- 5. Final Type Casting ---
    # My Thought Process: After filling NaNs, we can safely cast columns to integers
    # where appropriate (e.g., 'year_built'). This ensures data hygiene.
    int_cols = ['property_id', 'zip_code', 'year_built', 'days_on_mls', 'beds', 'full_baths']
    for col in int_cols:
        df_clean[col] = df_clean[col].astype(int)
        
    # --- 6. Final Formatting ---
    # Convert datetime back to string for JSON compatibility, handling NaT properly.
    df_clean['list_date'] = df_clean['list_date'].dt.strftime('%Y-%m-%d').replace({pd.NaT: None})

    # Return only the columns defined in our Pydantic schema.
    return df_clean[required_schema_columns]

def run_scoring_process():
    """
    The core logic for scoring properties. Designed to be run in the background.
    This function is FAILSAFE: it checks which properties are already in the DB
    and only processes the new ones.
    """
    print("Starting background scoring process...")
    db: Session = next(get_db()) # Get a new DB session for this background task
    try:
        time.sleep(4) 
        # 1. Get IDs of properties already in our database
        scored_ids = {res[0] for res in db.query(models.ScoredProperty.property_id).all()}
        print(f"Found {len(scored_ids)} properties already scored in the database.")

        # 2. Load and clean the for-sale properties
        forsale_df = pd.read_csv(settings.FOR_SALE_PROPERTIES_CSV)
        forsale_df_clean = clean_for_sale_data(forsale_df)

        # 3. Filter out properties that have already been scored
        new_properties_df = forsale_df_clean[~forsale_df_clean['property_id'].isin(scored_ids)]
        print(f"Found {len(new_properties_df)} new properties to score.")

        if new_properties_df.empty:
            print("No new properties to score. Process finished.")
            return

        # 4. Score new properties and add them to the database session
        for _, row in new_properties_df.iterrows():
            property_input = schemas.PropertyDataInput(**row.to_dict())
            scored_output = scoring_logic.score_property(property_input, data_handler.market_data_store)
            
            # Convert Pydantic output to SQLAlchemy model instance
            db_property = models.ScoredProperty(**scored_output.dict())
            db.add(db_property)

        # 5. Commit all new properties to the database in one transaction
        db.commit()
        print(f"Successfully scored and saved {len(new_properties_df)} new properties.")
    
    except Exception as e:
        print(f"An error occurred during scoring: {e}")
        db.rollback() # Roll back any partial changes if an error occurs
    finally:
        db.close() # Ensure the session is closed

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/process-scores", status_code=202)
async def trigger_scoring(background_tasks: BackgroundTasks):
    """
    Triggers the background task to score properties.
    It immediately returns a response while the processing happens in the background.
    Prevents starting a new task if one is already running.
    """
    if task_status['is_scoring']:
        raise HTTPException(status_code=409, detail="A scoring process is already in progress.")
    background_tasks.add_task(run_scoring_process)
    return {"message": "Property scoring process has been started in the background."}


@app.get("/properties", response_model=List[schemas.ScoreOutput])
async def get_scored_properties(
    db: Session = Depends(get_db),
    min_roi: Optional[float] = Query(None, description="Filter by minimum ROI percentage"),
    max_price: Optional[float] = Query(None, description="Filter by maximum list price"),
    min_beds: Optional[int] = Query(None, description="Filter by minimum number of bedrooms")
):
    """
    Retrieves scored properties from the database, with optional filters.
    This is the primary endpoint for the V3 web page.
    """
    query = db.query(models.ScoredProperty)

    if min_roi is not None:
        query = query.filter(models.ScoredProperty.roi_percentage >= min_roi)
    if max_price is not None:
        query = query.filter(models.ScoredProperty.list_price <= max_price)
    
    # Note: To filter by beds, we would need to add it to our model and schema.
    # For now, we focus on the core ROI and price filters.
    
    results = query.order_by(models.ScoredProperty.roi_percentage.desc()).all()
    return results

@app.get("/scoring-status")
async def get_scoring_status():
    """
    New endpoint for the frontend to poll.
    Returns the current status of the background task.
    """
    return {"is_scoring": task_status['is_scoring']}