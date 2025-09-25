# main.py
"""
The main FastAPI application file.
Handles API endpoints, background tasks, and database interactions.
"""
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import re, numpy as np

from config import settings
import models, schemas, scoring_logic, data_handler
from database import engine, get_db
from preprocessing import DataProcessor

models.Base.metadata.create_all(bind=engine)

# In-memory status tracker for the background task
task_status = {"is_scoring": False}

app = FastAPI(
    title="Flippit Real Estate Deal Scorer",
    description="An API to score and serve real estate deals using a V2 machine learning model.",
    version="2.0.0"
)

# Configure CORS to allow your React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load the trained ML model into memory when the application starts."""
    data_handler.load_resale_model()

def run_scoring_process():
    """The background task that scores new properties and saves them to the DB."""
    global task_status
    if task_status['is_scoring']: return
    task_status['is_scoring'] = True
    
    print("Starting background scoring process...")
    db: Session = next(get_db())
    processor = data_handler.model_store.get('processor')
    if not processor:
        print("ERROR: DataProcessor not loaded. Aborting scoring task.")
        task_status['is_scoring'] = False
        db.close()
        return
    
    try:
        scored_ids = {res[0] for res in db.query(models.ScoredProperty.property_id).all()}
        forsale_df = pd.read_csv(settings.FOR_SALE_PROPERTIES_CSV)

        # Simply deduplicate
        forsale_df['list_date'] = pd.to_datetime(forsale_df['list_date'], errors='coerce')
        forsale_df.sort_values(by='list_date', ascending=False, inplace=True)
        forsale_df.drop_duplicates(subset=['property_id'], keep='first', inplace=True)
        
        new_properties_df = forsale_df[~forsale_df['property_id'].isin(scored_ids)]
        
        if new_properties_df.empty:
            print("No new properties to score.")
            task_status['is_scoring'] = False
            db.close()
            return

        for _, row in new_properties_df.iterrows():
            try:
                # Let Pydantic handle initial parsing and default values from schema
                property_dict = row.to_dict()
                
                # Convert NaNs to None (JSON/Pydantic friendly) and handle data types
                for key, value in property_dict.items():
                    if pd.isna(value):
                        property_dict[key] = None # Pydantic handles Optional[type] = None

                # Specifically fix the list_date format if it's not None
                if property_dict.get('list_date'):
                    property_dict['list_date'] = property_dict['list_date'].strftime('%Y-%m-%d')

                # Ensure integer fields are integers, defaulting to 0 if None
                for key in ['beds', 'full_baths', 'half_baths', 'stories', 'parking_garage', 'days_on_mls']:
                    property_dict[key] = int(property_dict.get(key) or 0)

                # Ensure float fields are floats, defaulting to 0.0 if None
                for key in ['list_price', 'sqft', 'hoa_fee', 'tax', 'estimated_value', 'lot_sqft', 'latitude', 'longitude']:
                     property_dict[key] = float(property_dict.get(key) or 0.0)

                # 3. Create the Pydantic model from the CLEANED dictionary
                property_input = schemas.PropertyDataInput(**property_dict)
                
                # The scoring logic now handles all complex transformations internally
                scored_output = scoring_logic.score_property(property_input, data_handler.model_store)
                
                db_property = models.ScoredProperty(**scored_output.model_dump())
                db.add(db_property)
            except Exception as e:
                print(f"Could not score property {row.get('property_id')}: {e}")
            
        db.commit()
        print(f"Successfully scored and saved {len(new_properties_df)} new properties.")
    except Exception as e:
        print(f"An error occurred during scoring: {e}")
        db.rollback()
    finally:
        db.close()
        task_status['is_scoring'] = False
        print("Scoring process finished.")

# --- API ENDPOINTS ---

@app.post("/process-scores", status_code=202)
async def trigger_scoring(background_tasks: BackgroundTasks):
    """Triggers the background task to score new properties from the CSV."""
    if task_status['is_scoring']:
        raise HTTPException(status_code=409, detail="A scoring process is already running.")
    background_tasks.add_task(run_scoring_process)
    return {"message": "Property scoring process started."}

@app.get("/scoring-status")
async def get_scoring_status():
    """Endpoint for the frontend to check if the background task is running."""
    return task_status

@app.get("/properties", response_model=List[schemas.ScoreOutput])
async def get_scored_properties(
    db: Session = Depends(get_db),
    min_roi: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_beds: Optional[int] = Query(None, ge=0)
):
    """
    Retrieves scored properties from the database, with optional filters.
    This is the primary endpoint for the frontend web page.
    """
    query = db.query(models.ScoredProperty)
    if min_roi is not None:
        query = query.filter(models.ScoredProperty.roi_percentage >= min_roi)
    if max_price is not None:
        query = query.filter(models.ScoredProperty.list_price <= max_price)
    # The beds filter requires joining with the source data, a V3 enhancement
    # if min_beds is not None:
    #     query = query.filter(models.ScoredProperty.beds >= min_beds)
    
    return query.order_by(models.ScoredProperty.roi_percentage.desc()).all()