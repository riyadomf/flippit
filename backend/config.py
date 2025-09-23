# config.py
import os

class Settings:
    # --- Database Configuration ---
    # The URL for SQLite database file.
    DATABASE_URL: str = "sqlite:///./db/flippit.db"

    # --- Data File Paths ---
    # define the directory and filenames.
    DATA_DIR: str = "dataset"
    SOLD_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "sold_properties.csv")
    FOR_SALE_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "for_sale_properties.csv")

# Create a single instance of the settings to be imported by other modules.
settings = Settings()