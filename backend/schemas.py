# schemas.py
"""
Defines the Pydantic models for data validation and serialization.
These models define the shape of the data for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
import datetime


class PropertyDataInput(BaseModel):
    # Core Property Features
    property_id: int
    list_price: float
    sqft: float
    zip_code: int
    year_built: int
    beds: Optional[int] = 0
    full_baths: Optional[int] = 0
    half_baths: Optional[float] = 0.0
    stories: Optional[float] = 1.0
    parking_garage: Optional[float] = 0.0
    lot_sqft: Optional[float] = 0.0
    neighborhoods: Optional[str] = "Unknown"
    hoa_fee: Optional[float] = 0.0
    tax: Optional[float] = 0.0
    days_on_mls: Optional[int] = 0
    estimated_value: Optional[float] = 0.0
    
    # Address and Photo info (for output)
    full_street_line: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    primary_photo: Optional[str] = None
    list_date: Optional[str] = None
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    
    # --- NEW: Pre-generated LLM Features ---
    # My Thought Process: The API now expects this data to be provided,
    # as it will be read from the enriched CSV file. We provide safe defaults.
    renovation_level: str = "Unknown"
    llm_quality_score: int = 5 # Default to a neutral score
    llm_risk_score: int = 5    # Default to a neutral score
    positive_features: Optional[List[str]] = []
    negative_features: Optional[List[str]] = []



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


class LLMAnalysisOutput(BaseModel):
    """Schema to validate the structured JSON output from the LLM."""
    renovation_level: str = Field(..., description="One of ['Cosmetic', 'Medium', 'Heavy', 'Gut', 'Unknown']")
    quality_score: int = Field(..., description="A 1-10 score of the property's finishes, appeal, and luxury.")
    risk_score: int = Field(..., description="A 1-10 score of potential hidden problems or red flags.")
    positive_features: List[str]
    negative_features: List[str]