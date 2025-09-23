# scoring_logic.py

from schemas import PropertyDataInput, ScoreOutput, ScoreResultBase
import re

# ==============================================================================
# V1.0 CALCULATION MODULES
# ==============================================================================

def estimate_resale_price(sqft: float, zip_code: int, market_data: dict) -> float:
    """
    V1: Estimates resale price based on ZIP code-level median price per square foot.
    """
    baseline_price_per_sqft = market_data['median_by_zip'].get(zip_code, market_data['overall_median'])
    return sqft * baseline_price_per_sqft

def estimate_renovation_cost(sqft: float, year_built: int, description: str) -> float:
    """
    V1: Estimates renovation costs using a heuristic tiered model based on age and keywords.
    """
    description = str(description).lower()
    renovation_cost_per_sqft = 45  # Default: Medium renovation

    high_cost_keywords = ['fixer-upper', 'tlc', 'as-is', 'rehab', 'investor special', 'complete rehab']
    low_cost_keywords = ['fully renovated', 'newly updated', 'move-in ready', 'new kitchen']
    
    if any(keyword in description for keyword in high_cost_keywords):
        renovation_cost_per_sqft = 75
    elif year_built < 1960:
        renovation_cost_per_sqft = 70
    elif any(keyword in description for keyword in low_cost_keywords):
        renovation_cost_per_sqft = 20
        
    return sqft * renovation_cost_per_sqft

def calculate_other_costs(estimated_resale_price: float, tax_yearly: float, hoa_fee: float) -> float:
    """
    V1: Calculates carrying and selling costs based on fixed assumptions.
    """
    # Assumption: 6% for realtor commissions, closing costs, etc.
    selling_costs = estimated_resale_price * 0.06 
    
    # Assumption: 4 month holding period
    monthly_taxes = tax_yearly / 12 if tax_yearly else 0
    carrying_costs = 4 * (monthly_taxes + hoa_fee)
    
    return selling_costs + carrying_costs

def assess_risk(year_built: int, days_on_mls: int, list_price: float, estimated_value: float) -> int:
    """
    V1: Calculates a simple risk score (0-10) based on property attributes.
    """
    risk_score = 0
    if year_built < 1950:
        risk_score += 3
    if days_on_mls > 60:
        risk_score += 2
    # Check if estimated_value is valid before using it
    if estimated_value and estimated_value > 0 and list_price < (estimated_value * 0.8):
        risk_score += 3
    return min(risk_score, 10)

def assign_grade(roi: float, risk_score: int) -> str:
    """
    V1: Assigns a letter grade based on ROI and risk.
    """
    if roi > 25 and risk_score < 4: return 'A'
    if roi > 20 and risk_score < 6: return 'B'
    if roi > 15 and risk_score < 8: return 'C'
    if roi > 10: return 'D'
    return 'F'

# ==============================================================================
# ORCHESTRATOR
# This main function calls the modular pieces above to generate the final score.
# ==============================================================================

def score_property(property_data: PropertyDataInput, market_data: dict) -> ScoreResultBase:
    """
    Orchestrates the scoring process and returns a result object without a database ID.
    """
    resale_price = estimate_resale_price(property_data.sqft, property_data.zip_code, market_data)
    reno_cost = estimate_renovation_cost(property_data.sqft, property_data.year_built, property_data.text)
    other_costs = calculate_other_costs(resale_price, property_data.tax, property_data.hoa_fee)
    total_costs = property_data.list_price + reno_cost + other_costs
    expected_profit = resale_price - total_costs
    total_cash_spent = property_data.list_price + reno_cost
    roi = (expected_profit / total_cash_spent) * 100 if total_cash_spent > 0 else 0
    risk = assess_risk(property_data.year_built, property_data.days_on_mls, property_data.list_price, property_data.estimated_value)
    grade = assign_grade(roi, risk)
    explanation = f"Grade {grade} based on an estimated ROI of {roi:.1f}% and a risk score of {risk}/10. Resale value estimated at ${resale_price:,.0f}."

    # UPDATED: Returns the base schema object, which does not have an 'id' field.
    return ScoreResultBase(
        property_id=property_data.property_id,
        address=f"{property_data.full_street_line}, {property_data.city}, {property_data.state}",
        list_price=property_data.list_price,
        estimated_resale_price=resale_price,
        renovation_cost=reno_cost,
        carrying_and_selling_costs=other_costs,
        expected_profit=expected_profit,
        roi_percentage=roi,
        risk_score=risk,
        overall_grade=grade,
        explanation=explanation,
        latitude=property_data.latitude,
        longitude=property_data.longitude,
        primary_photo=property_data.primary_photo
    )