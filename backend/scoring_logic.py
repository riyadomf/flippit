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
        'Medium': 15.0,
        'Heavy': 35.0,
        'Gut': 55.0,
        'Unknown': 15.0
    }
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



def analyze_description_with_llm(text: str) -> LLMAnalysisOutput:
    """
    Sends the property description to the LLM and returns a validated,
    structured analysis. Includes robust error handling.
    """

    default_response = LLMAnalysisOutput(
        renovation_level="Unknown", positive_features=[], negative_features=[],
        estimated_risk_score=5, risk_factors=["LLM analysis failed or text was empty."]
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


def estimate_renovation_cost(sqft: float, llm_analysis: LLMAnalysisOutput) -> float:
    """Calculates renovation cost based on the LLM's assessment."""
    cost_per_sqft = ScoringConstants.RENOVATION_COST_MAP.get(llm_analysis.renovation_level, 15.0)
    return sqft * cost_per_sqft



def calculate_other_costs(resale_price: float, tax: float, hoa: float) -> float:
    """Calculates carrying and selling costs based on tuned assumptions."""
    consts = ScoringConstants
    selling_costs = resale_price * consts.SELLING_COST_PERCENTAGE
    monthly_taxes = tax / 12 if tax else 0
    carrying_costs = consts.HOLDING_PERIOD_MONTHS * (monthly_taxes + hoa)
    return selling_costs + carrying_costs

def assess_risk(property_data: PropertyDataInput, llm_analysis: LLMAnalysisOutput) -> int:
    """Calculates a hybrid risk score from the LLM and structured data."""
    # Start with the LLM's nuanced textual risk assessment
    risk = llm_analysis.estimated_risk_score
    
    # Add points for structural factors the LLM can't see
    if property_data.year_built < 1960: risk += 2
    if property_data.days_on_mls > 90: risk += 1
    if property_data.list_price < (property_data.estimated_value * 0.75): risk += 2

    return min(risk, 10)




def assign_grade(roi: float, risk: int) -> str:
    """Assigns a letter grade based on tuned ROI and risk thresholds."""
    if roi > 18 and risk < 5: return 'A'
    if roi > 15 and risk < 7: return 'B'
    if roi > 12 and risk < 8: return 'C'
    if roi > 8: return 'D'
    return 'F'



def generate_explanation(grade: str, roi: float, arv: float, llm_analysis: LLMAnalysisOutput) -> str:
    """Creates a rich, dynamic explanation using the LLM's findings."""
    base = (f"Grade {grade} based on an estimated ROI of {roi:.1f}%. "
            f"ARV model predicts a resale value of ${arv:,.0f}. ")
    
    reno_reason = f"Renovation is estimated as '{llm_analysis.renovation_level}'."
    
    pos_features = ", ".join(llm_analysis.positive_features)
    neg_features = ", ".join(llm_analysis.negative_features)
    
    details = ""
    if pos_features:
        details += f" Key positives include: {pos_features}."
    if neg_features:
        details += f" Areas of concern are: {neg_features}."
        
    return base + reno_reason + details


# --- Orchestrator ---

def score_property(property_data: PropertyDataInput, model_store: dict) -> ScoreResultBase:
    """
    V3 (Hybrid): Orchestrates the full scoring process using the XGBoost ARV model
    and the LLM for text analysis.
    """
    # 1. Get ARV from the V2 XGBoost model (numerical analysis)
    arv_prediction = estimate_resale_price(property_data, model_store)
    
    # 2. Get nuanced analysis from the V3 LLM (textual analysis)
    llm_analysis = analyze_description_with_llm(property_data.text)
    
    # 3. Calculate final scores using the hybrid results
    reno_cost = estimate_renovation_cost(property_data.sqft, llm_analysis)
    risk = assess_risk(property_data, llm_analysis)
    other_costs = calculate_other_costs(arv_prediction, property_data.tax, property_data.hoa_fee)
    
    # 4. Final Calculations
    cash_invested = property_data.list_price + reno_cost
    profit = arv_prediction - (property_data.list_price + reno_cost + other_costs)
    roi = (profit / cash_invested) * 100 if cash_invested > 0 else 0
    grade = assign_grade(roi, risk) # Re-use the simple grading function
    explanation = generate_explanation(grade, roi, arv_prediction, llm_analysis)
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