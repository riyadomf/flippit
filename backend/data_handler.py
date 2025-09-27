# data_handler.py
"""
Handles loading of pre-trained models and other data assets at application startup.
"""
import joblib
from config import settings

# This dictionary will hold our trained ML model in memory.
model_store = {}

def load_resale_model():
    """Loads the pre-trained model and the fitted DataProcessor."""
    try:
        payload = joblib.load(settings.MODEL_PATH)
        model_store['model'] = payload['model']
        model_store['processor'] = payload['processor'] # Load the processor
        print("resale prediction model and data processor loaded successfully.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model file '{settings.MODEL_PATH}' not found.")
        model_store['model'] = None
        model_store['processor'] = None