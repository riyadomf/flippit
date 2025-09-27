# train_model.py
"""
This standalone script handles the entire model training process.
Run this file from your terminal to create the resale_model.pkl file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

from preprocessing import DataProcessor
from config import settings

def train_resale_model():
    """
    Trains the model on the full dataset, but evaluates it specifically
    on the "gold standard" subset to get a true measure of ARV prediction accuracy.
    """
    print("--- Starting Model Training Pipeline ---")

    # --- 1. Load the Enriched Data ---
    try:
        raw_sold_df = pd.read_csv(settings.ENRICHED_SOLD_PROPERTIES_CSV)
    except FileNotFoundError:
        print(f"ERROR: Enriched data not found at {settings.ENRICHED_SOLD_PROPERTIES_CSV}")
        print("Please run 'python generate_llm_features.py sold' first.")
        return
    
    # 1. Create and Fit the DataProcessor on the entire raw dataset
    processor = DataProcessor()
    processor.fit(raw_sold_df)

    # 2. Transform the entire dataset to get the full feature matrix
    X, y = processor.transform(raw_sold_df)
    print(f"Full dataset transformed. Feature matrix has {X.shape[1]} features.")

    # 3. Strict Train-Test Split BEFORE any filtering
    # This creates a "lockbox" holdout set that the model will never see during training.
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} for training pool and {len(X_holdout)} for holdout pool.")

    # A renovated home should have a low risk score and a high quality score from the LLM
    # Ensure the target column for filtering exists
    gold_standard_filter_col = 'renovation_level_Cosmetic'
    gold_standard_indices = X_holdout[
        (X_holdout['llm_risk_score'] <= 2) &
        (X_holdout['llm_quality_score'] >= 6) &
        (X_holdout[gold_standard_filter_col] == 1)  # <-- THE CORRECTED LOGIC
    ].index
    
    X_test_final = X_holdout.loc[gold_standard_indices]
    y_test_final = y_holdout.loc[gold_standard_indices]


    # Fallback if no ARV-like properties are found
    if len(X_test_final) < 10:
        print("WARNING: No 'gold standard' properties found in the holdout set for evaluation.")
        print("WARNING: Too few 'gold standard' properties found. Evaluating on general test set instead.")
        X_test_final, y_test_final = X_holdout, y_holdout
    
    print(f"Model will train on {len(X_train)} properties.")
    print(f"Model will be evaluated on {len(X_test_final)} unseen 'gold standard' properties.")


    # 5. Define, Train, and Evaluate
    model = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.03, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1, 
        subsample=0.7, 
        colsample_bytree=0.7,
        early_stopping_rounds=50)
    
    # Train ONLY on the training set
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test_final, y_test_final)], 
        verbose=False
    )
    
    # Evaluate ONLY on the gold standard test set
    predictions = model.predict(X_test_final)
    mae = mean_absolute_error(y_test_final, predictions)
    print(f"  > Mean Absolute Error (MAE) on 'Gold Standard' Test Set: ${mae:,.2f}")
    

    # 4. Final Evaluation on the Gold Standard Set
    print("\n--- Gold Standard Performance ---")
    y_pred_gold = model.predict(X_test_final)
    mae_gold = mean_absolute_error(y_test_final, y_pred_gold)
    r2_gold = r2_score(y_test_final, y_pred_gold)
    print(f"Gold Standard MAE: ${mae_gold:,.2f}")
    print(f"Gold Standard R-squared: {r2_gold:.4f}")

    # 6. Save the final payload
    model_payload = {'model': model, 'processor': processor}
    joblib.dump(model_payload, settings.MODEL_PATH)
    print(f"--- Trained model and fitted processor saved to '{settings.MODEL_PATH}' ---")

if __name__ == "__main__":
    train_resale_model()