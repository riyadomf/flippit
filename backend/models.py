# models.py
from sqlalchemy import Column, Integer, String, Float
from database import Base

class ScoredProperty(Base):
    __tablename__ = "scored_properties"

    id = Column(Integer, primary_key=True, index=True)
    property_id = Column(Integer, unique=True, index=True)
    address = Column(String, index=True)
    list_price = Column(Float)
    estimated_resale_price = Column(Float)
    renovation_cost = Column(Float)
    carrying_and_selling_costs = Column(Float)
    expected_profit = Column(Float)
    roi_percentage = Column(Float)
    risk_score = Column(Integer)
    overall_grade = Column(String)
    explanation = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    primary_photo = Column(String)