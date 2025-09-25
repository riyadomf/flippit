# train_model.py
"""
This standalone script handles the entire model training process.
Run this file from your terminal to create the resale_model.pkl file.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

from preprocessing import preprocess_for_resale_model
from config import settings

def train_resale_model():
    """
    Full pipeline to load, preprocess, train, evaluate, and save the resale price model.
    """
    print("--- Starting V2 Model Training Pipeline ---")
    
    # 1. Load and Preprocess Data
    raw_sold_df = pd.read_csv(settings.SOLD_PROPERTIES_CSV)
    X, y = preprocess_for_resale_model(raw_sold_df)
    
    # 2. Split Data for Training and Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
    
    # 3. Define and Train the Model
    print("\nTraining XGBoost Regressor model...")
    model = XGBRegressor(
        n_estimators=500,     # Number of trees to build.
        learning_rate=0.05,   # How much to shrink the contribution of each tree.
        max_depth=5,          # Maximum depth of a tree.
        subsample=0.8,        # Fraction of samples to be used for fitting each tree.
        colsample_bytree=0.8, # Fraction of features to be used for fitting each tree.
        random_state=42,
        n_jobs=-1             # Use all available CPU cores
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # 4. Evaluate the Model
    # My Thought Process: We MUST evaluate the model on the unseen test data.
    # This tells us the real-world expected error.
    print("\nEvaluating model performance...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"  > Mean Absolute Error (MAE) on Test Set: ${mae:,.2f}")
    print("  > This means, on average, our model's price prediction is off by this amount.")
    
    # 5. Save the Trained Model and the Columns
    model_payload = {
        'model': model,
        'training_columns': X_train.columns.tolist()
    }
    
    joblib.dump(model_payload, settings.MODEL_PATH)
    print(f"--- Trained model saved to '{settings.MODEL_PATH}' ---")

if __name__ == "__main__":
    train_resale_model()