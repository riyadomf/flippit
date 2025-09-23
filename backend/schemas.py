# schemas.py
from pydantic import BaseModel
from typing import Optional

class PropertyDataInput(BaseModel):
    property_id: int
    list_price: float
    list_date: Optional[str] = None
    sqft: float
    zip_code: int
    year_built: int
    hoa_fee: Optional[float] = 0.0
    tax: Optional[float] = 0.0
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
    primary_photo: Optional[str] = ""
    

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
    id: int
    
    class Config:
        orm_mode = True