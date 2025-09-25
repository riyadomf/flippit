# schemas.py
"""
Defines the Pydantic models for data validation and serialization.
These models define the shape of the data for API requests and responses.
"""
from pydantic import BaseModel
from typing import Optional
import datetime


class PropertyDataInput(BaseModel):
    """
    V2 Schema: Includes ALL raw features needed for preprocessing before prediction.
    """
    property_id: int
    list_price: float
    sqft: float
    zip_code: int
    year_built: int
    
    # Core features now required by the V2 model
    beds: Optional[int] = 0
    full_baths: Optional[int] = 0
    half_baths: Optional[int] = 0
    lot_sqft: Optional[float] = 0.0
    neighborhoods: Optional[str] = "Unknown" # Provide a safe default
    stories: Optional[int] = 1 # Provide a safe default
    parking_garage: Optional[int] = 0 # Provide a safe default
    
    # Features for other scoring modules
    hoa_fee: Optional[float] = 0.0
    tax: Optional[float] = 0.0
    text: Optional[str] = ""
    estimated_value: Optional[float] = 0.0
    days_on_mls: Optional[int] = 0
    
    # Pass-through features for the final output
    full_street_line: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    primary_photo: Optional[str] = None
    list_date: Optional[str] = None
    

class ScoreResultBase(BaseModel):
    property_id: int
    address: str
    list_price: float
    estimated_resale_price: float
    renovation_cost: float
    carrying_and_selling_costs: float
    expected_profit: float
    roi_percentage: float
    risk_score: int
    overall_grade: str
    explanation: str
    latitude: float
    longitude: float
    primary_photo: Optional[str] = ""


class ScoreOutput(ScoreResultBase):
    """Final output schema including database fields, for sending data to the client."""
    id: int
    created_at: datetime.datetime
    
    class Config:
        orm_mode = True # Enables the model to be created from an ORM object