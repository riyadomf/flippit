# scoring_logic.py
"""
Contains all the modular functions for calculating the different components
of a property's flip score.
"""
import pandas as pd
from schemas import PropertyDataInput, ScoreResultBase
from config import settings 


# --- Tunable Constants for Scoring Logic ---
class ScoringConstants:
    HIGH_COST_SQFT = 45
    MEDIUM_COST_SQFT = 25
    COSMETIC_COST_SQFT = 12
    LOW_COST_SQFT = 5
    SELLING_COST_PERCENTAGE = 0.05
    HOLDING_PERIOD_MONTHS = 3


# --- Calculation Modules ---

def estimate_resale_price(property_data: PropertyDataInput, model_store: dict) -> float:
    """Estimates resale price using the pre-trained model and processor."""
    model = model_store.get('model')
    processor = model_store.get('processor')
    if not model or not processor: return 0.0

    property_df = pd.DataFrame([property_data.model_dump()])

    # The processor now handles all data preparation for the model
    X_processed, _ = processor.transform(property_df)


    # Manually override the condition flags in the FINAL feature matrix.
    # Find the final one-hot encoded columns and set them to represent
    # the ideal "renovated" state. This asks the model a clear question:
    # "What would the price of this property be IF it were renovated?"
    
    if 'is_fixer_upper' in X_processed.columns:
        X_processed['is_fixer_upper'] = 0
    
    if 'is_renovated' in X_processed.columns:
        X_processed['is_renovated'] = 1
    
    prediction = model.predict(X_processed)
    return float(prediction[0])



def estimate_renovation_cost(sqft: float, year_built: int, description: str) -> float:
    """
    Heuristics: Uses both positive and negative keywords to
    determine the renovation cost tier.
    """
    description = str(description).lower()
    consts = ScoringConstants
    
    # This model now uses keywords for both high AND low cost.
    high_cost_keywords = settings.HIGH_COST_KEYWORDS
    low_cost_keywords = settings.LOW_COST_KEYWORDS
    
    if any(k in description for k in high_cost_keywords):
        cost_per_sqft = consts.HIGH_COST_SQFT
    elif any(k in description for k in low_cost_keywords):
        cost_per_sqft = consts.LOW_COST_SQFT
    elif year_built < 1965:
        cost_per_sqft = consts.MEDIUM_COST_SQFT
    else:
        cost_per_sqft = consts.COSMETIC_COST_SQFT
        
    return sqft * cost_per_sqft

def calculate_other_costs(resale_price: float, tax: float, hoa: float) -> float:
    """Calculates carrying and selling costs based on tuned assumptions."""
    consts = ScoringConstants
    selling_costs = resale_price * consts.SELLING_COST_PERCENTAGE
    monthly_taxes = tax / 12 if tax else 0
    carrying_costs = consts.HOLDING_PERIOD_MONTHS * (monthly_taxes + hoa)
    return selling_costs + carrying_costs

def assess_risk(year_built: int, days_on_mls: int, list_price: float, est_value: float) -> int:
    """Calculates a simple risk score based on property attributes."""
    risk = 0
    if year_built < 1950: risk += 3
    if days_on_mls > 60: risk += 2
    if est_value and est_value > 0 and list_price < (est_value * 0.8): risk += 3
    return min(risk, 10)

def assign_grade(roi: float, risk: int) -> str:
    """Assigns a letter grade based on tuned ROI and risk thresholds."""
    if roi > 18 and risk < 5: return 'A'
    if roi > 15 and risk < 7: return 'B'
    if roi > 12 and risk < 8: return 'C'
    if roi > 8: return 'D'
    return 'F'


# --- Orchestrator ---

def score_property(property_data: PropertyDataInput, model_store: dict) -> ScoreResultBase:
    """Orchestrates the full scoring process using the resale model."""
    resale_price = estimate_resale_price(property_data, model_store)
    reno_cost = estimate_renovation_cost(property_data.sqft, property_data.year_built, property_data.text)
    other_costs = calculate_other_costs(resale_price, property_data.tax, property_data.hoa_fee)
    
    total_costs = property_data.list_price + reno_cost + other_costs
    profit = resale_price - total_costs
    cash_invested = property_data.list_price + reno_cost
    roi = (profit / cash_invested) * 100 if cash_invested > 0 else 0
    
    risk = assess_risk(property_data.year_built, property_data.days_on_mls, property_data.list_price, property_data.estimated_value)
    grade = assign_grade(roi, risk)
    
    explanation = (f"Grade {grade} based on an estimated ROI of {roi:.1f}% and a risk score of {risk}/10. "
                   f"Resale value estimated by the model at ${resale_price:,.0f}.")

    return ScoreResultBase(
        property_id=property_data.property_id,
        address=f"{property_data.full_street_line}, {property_data.city}, {property_data.state}",
        list_price=property_data.list_price,
        estimated_resale_price=resale_price,
        renovation_cost=reno_cost,
        carrying_and_selling_costs=other_costs,
        expected_profit=profit,
        roi_percentage=roi,
        risk_score=risk,
        overall_grade=grade,
        explanation=explanation,
        latitude=property_data.latitude,
        longitude=property_data.longitude,
        primary_photo=property_data.primary_photo
    )