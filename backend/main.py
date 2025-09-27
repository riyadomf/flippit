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
    
    # Define batch size
    BATCH_SIZE = 10 
    

    # Get a new DB session for the entire process duration
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
        
        new_properties_df = forsale_df[~forsale_df['property_id'].isin(scored_ids)]
        
        if new_properties_df.empty:
            print("No new properties to score.")
            task_status['is_scoring'] = False
            db.close()
            return
        
        # Use the processor to get a list of clean, Pydantic-ready dictionaries
        clean_property_dicts = processor.prepare_inference_data(new_properties_df)
        print(f"Found and cleaned {len(clean_property_dicts)} new properties to score.")

        total_to_score = len(clean_property_dicts)
        if total_to_score == 0:
            print("No new properties to score.")
            return
        
        print(f"Found {total_to_score} new properties to score. Processing in batches of {BATCH_SIZE}.")


        for i in range(0, total_to_score, BATCH_SIZE):
            batch = clean_property_dicts[i:i + BATCH_SIZE]
            print(f"--- Processing batch {i//BATCH_SIZE + 1} ({len(batch)} properties) ---")
            
            # This inner loop is for a single batch
            for prop_dict in batch:
                try:
                    # Create the Pydantic model from the clean dictionary
                    property_input = schemas.PropertyDataInput(**prop_dict)
                    print(property_input)
                    print(f"Scoring property ID: {property_input.property_id}")
                    # Call the scoring logic
                    scored_output = scoring_logic.score_property(property_input, data_handler.model_store)
                    print(f"Scored property ID {property_input.property_id}: ROI {scored_output.roi_percentage:.2f}")
                    
                    # Add the result to the current session's transaction
                    db_property = models.ScoredProperty(**scored_output.dict())
                    db.add(db_property)
                    
                except Exception as e:
                    # If a single property fails, log it and continue with the rest of the batch
                    print(f"ERROR: Could not score property {prop_dict.get('property_id')}: {e}")
            
            # Commit the transaction for the CURRENT BATCH
            # We commit after
            # every 10 properties. If the script crashes during the next batch,
            # this batch's work is already saved permanently.
            try:
                db.commit()
                print(f"--- Batch {i//BATCH_SIZE + 1} successfully committed to the database. ---")
            except Exception as e:
                print(f"DATABASE ERROR: Could not commit batch {i//BATCH_SIZE + 1}: {e}")
                db.rollback() # Rollback the failed batch and continue to the next one

    except Exception as e:
        # This catches errors during the initial data loading phase
        print(f"A critical error occurred during data preparation: {e}")
    finally:
        # Always ensure the session is closed and status is reset
        db.close()
        task_status['is_scoring'] = False
        print("--- Scoring process finished. ---")

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