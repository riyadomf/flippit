# schemas.py

from pydantic import BaseModel, Field
from typing import Optional

# This model defines the structure of the JSON we expect to receive in a POST request.
# It ensures that the incoming data is validated before our logic even runs.
class PropertyDataInput(BaseModel):
    property_id: int
    list_price: float
    sqft: float
    zip_code: int
    year_built: int
    hoa_fee: Optional[float] = 0.0 # Optional field with a default value
    tax: Optional[float] = 0.0 # Using the cleaned 'tax' value
    text: Optional[str] = ""
    estimated_value: Optional[float] = 0.0
    days_on_mls: Optional[int] = 0
    full_street_line: Optional[str] = ""
    city: Optional[str] = ""
    state: Optional[str] = ""
    beds: Optional[int] = 0
    full_baths: Optional[int] = 0
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    
# This model defines the structure of the JSON response our API will send back.
class ScoreOutput(BaseModel):
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