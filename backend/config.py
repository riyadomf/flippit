# config.py
import os

class Settings:
    """
    Central configuration file to store application-wide settings.
    """
    DATABASE_URL: str = "sqlite:///./data/flippit.db"
    DATA_DIR: str = "backend/data"
    SOLD_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "sold_properties.csv")
    FOR_SALE_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "for_sale_properties.csv")
    MODEL_PATH: str = "resale_model.pkl"

# Create a single instance of the settings to be imported by other modules.
settings = Settings()