# scoring_logic.py
"""
Contains all the modular functions for calculating the different components
of a property's flip score.
"""
import pandas as pd
import json
import ollama
from schemas import PropertyDataInput, ScoreResultBase, LLMAnalysisOutput
from config import settings 
from prompts import FLIP_ANALYSIS_PROMPT_TEMPLATE

# --- Tunable Constants for Scoring Logic ---
class ScoringConstants:
    # HIGH_COST_SQFT = 45
    # MEDIUM_COST_SQFT = 25
    # COSMETIC_COST_SQFT = 12
    # LOW_COST_SQFT = 5

    RENOVATION_COST_MAP = {
        'Cosmetic': 5.0,
        'Medium': 20.0,
        'Heavy': 45.0,
        'Gut': 55.0,
        'Unknown': 15.0
    }
    SELLING_COST_PERCENTAGE = 0.05
    HOLDING_PERIOD_MONTHS = 3


def analyze_description_with_llm(text: str) -> LLMAnalysisOutput:
    """
    Sends the property description to the LLM and returns a validated,
    structured analysis. Includes robust error handling.
    """

    default_response = LLMAnalysisOutput(
        renovation_level="Unknown", positive_features=[], negative_features=[],
        quality_score=4, risk_score=3
    )
    
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return default_response

    try:
        prompt = FLIP_ANALYSIS_PROMPT_TEMPLATE.format(description=text)
        response = ollama.chat(
            model=settings.OLLAMA_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # This is a crucial instruction!
        )
        
        # The Ollama library gives the parsed JSON content
        llm_dict = json.loads(response['message']['content'])
        validated_output = LLMAnalysisOutput(**llm_dict)
        return validated_output

    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM analysis failed: {e}")
        return default_response

# --- Calculation Modules ---



def estimate_resale_price(property_data: PropertyDataInput, model_store: dict) -> float:
    """
    V2.2: Estimates resale price by simulating a high-quality renovation state
    using the LLM's 'quality_score'.
    """
    model = model_store.get('model')
    processor = model_store.get('processor')
    if not model or not processor: return 0.0

    property_df = pd.DataFrame([property_data.model_dump()])
    
    print(f"Transforming property ID: {property_data.property_id} for ARV prediction.")
    X_processed, _ = processor.transform(property_df)
    print(f"Feature matrix shape after processing: {X_processed.shape} for property ID: {property_data.property_id}")

    # --- Manually Override to Represent an "After Repair" State ---
    # A top-tier renovation should result in a high quality_score.
    # We set it to 9 (not 10, to be slightly conservative) and recalculate interactions.
    # This is a much stronger signal to the model than flipping a binary flag.
    if 'llm_quality_score' in X_processed.columns:
        X_processed['llm_quality_score'] = 10  # Simulate a high-quality, desirable finish
    if 'llm_quality_score' in X_processed.columns:
        X_processed['llm_risk_score'] = 1  # Simulate a high-quality, desirable finish
    
    # Recalculate the interaction features with the new quality score
    if 'sqft_x_quality_score' in X_processed.columns:
        X_processed['sqft_x_quality_score'] = X_processed['sqft'] * X_processed['llm_quality_score']
    

    # Set the one-hot encoded renovation level to our ideal state: "Cosmetic"
    reno_level_cols = [col for col in X_processed.columns if col.startswith('renovation_level_')]
    for col in reno_level_cols:
        X_processed[col] = 1 if col == 'renovation_level_Cosmetic' else 0

    print(f"Predicting resale price for property ID: {property_data.property_id}")
    print(f"Processed features: {X_processed.iloc[0].to_dict()}")
    prediction = model.predict(X_processed)
    return float(prediction[0])



def estimate_renovation_cost(sqft: float, renovation_level: str) -> float:
    """Calculates renovation cost based on the pre-generated 'renovation_level'."""
    cost_per_sqft = ScoringConstants.RENOVATION_COST_MAP.get(renovation_level, 15.0)
    return sqft * cost_per_sqft



def calculate_other_costs(resale_price: float, tax: float, hoa: float) -> float:
    """Calculates carrying and selling costs based on tuned assumptions."""
    consts = ScoringConstants
    selling_costs = resale_price * consts.SELLING_COST_PERCENTAGE
    monthly_taxes = tax / 12 if tax else 0
    carrying_costs = consts.HOLDING_PERIOD_MONTHS * (monthly_taxes + hoa)
    return selling_costs + carrying_costs


def assess_risk(property_data: PropertyDataInput) -> int:
    """Calculates a hybrid risk score from the pre-generated LLM score and structured data."""
    # Start with the pre-generated LLM score
    risk = property_data.llm_risk_score
    
    # Add points for structural factors
    if property_data.year_built < 1960: risk += 1
    if property_data.days_on_mls > 90: risk += 1
    if property_data.estimated_value and property_data.list_price < (property_data.estimated_value * 0.75): risk += 1
        
    return min(risk, 10)

def assign_grade(roi: float, risk: int) -> str:
    # ... (This function is unchanged) ...
    if roi > 18 and risk < 5: return 'A'
    if roi > 15 and risk < 7: return 'B'
    if roi > 12 and risk < 8: return 'C'
    if roi > 8: return 'D'
    return 'F'

def generate_explanation(grade: str, roi: float, arv: float, property_data: PropertyDataInput) -> str:
    """Creates a rich explanation using the pre-generated LLM findings."""
    base = (f"Grade {grade} based on an estimated ROI of {roi:.1f}%. "
            f"ARV model predicts a resale value of ${arv:,.0f}. ")
    
    reno_reason = f"Renovation is estimated as '{property_data.renovation_level}'."
    
    details = ""
    if property_data.positive_features:
        details += f" Key positives include: {', '.join(property_data.positive_features)}."
    if property_data.negative_features:
        details += f" Areas of concern are: {', '.join(property_data.negative_features)}."
        
    return base + reno_reason + details


# --- Orchestrator ---

def score_property(property_data: PropertyDataInput, model_store: dict) -> ScoreResultBase:
    """
    V2.3 (Optimized): Orchestrates scoring using pre-generated LLM features.
    NO LIVE LLM CALLS ARE MADE HERE.
    """
    # My Thought Process: This function is now much faster and simpler.
    # It just takes the clean PropertyDataInput and calls the calculation modules.

    print("--- Scoring Property ID:", property_data.property_id, "---")
    print(f"Property Data: {property_data}")
    
    # 1. Estimate ARV using the V2 model
    print(f"Estimating resale price for property ID: {property_data.property_id}")
    arv_prediction = estimate_resale_price(property_data, model_store)
    print(f"Predicted ARV: ${arv_prediction:,.2f} for property ID: {property_data.property_id}")
    
    # 2. Calculate other scores using the pre-generated LLM data
    reno_cost = estimate_renovation_cost(property_data.sqft, property_data.renovation_level)
    risk = assess_risk(property_data)
    other_costs = calculate_other_costs(arv_prediction, property_data.tax, property_data.hoa_fee)
    
    # 3. Final Calculations
    cash_invested = property_data.list_price + reno_cost
    profit = arv_prediction - (property_data.list_price + reno_cost + other_costs)
    roi = (profit / cash_invested) * 100 if cash_invested > 0 else 0
    grade = assign_grade(roi, risk)
    explanation = generate_explanation(grade, roi, arv_prediction, property_data)
    
    # 4. Return the final structured output
    return ScoreResultBase(
        property_id=property_data.property_id,
        address=f"{property_data.full_street_line}, {property_data.city}, {property_data.state}",
        list_price=property_data.list_price,
        estimated_resale_price=arv_prediction,
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