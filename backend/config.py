# config.py
import os

class Settings:
    """
    Central configuration file to store application-wide settings.
    """
    DATABASE_URL: str = "sqlite:///./data/flippit.db"
    DATA_DIR: str = "data"
    SOLD_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "sold_properties.csv")
    FOR_SALE_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "for_sale_properties.csv")

    ENRICHED_SOLD_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "llm_enriched_sold_properties.csv")
    ENRICHED_FOR_SALE_PROPERTIES_CSV: str = os.path.join(DATA_DIR, "llm_enriched_for_sale_properties.csv")
    MODEL_PATH: str = "resale_model.pkl"

    OLLAMA_MODEL_NAME: str = "llama3"

    # KEYWORD_MAP: dict = {
    #     # Premium finishes (for ARV model)
    #     'has_granite': 'granite',
    #     'has_hardwood': 'hardwood|wood floor',
    #     'has_stainless': 'stainless steel|stainless appliances',
    #     # Condition flags (for ARV model AND Renovation Cost model)
    #     'is_fixer_upper': '|'.join([
    #         'fixer', 'fixer-upper', 'tlc', 'as-is', 'as is', 'rehab',
    #         'investor special', 'needs work', 'handyman', 'opportunity'
    #     ]),
    #     'is_renovated': '|'.join([
    #         'fully renovated', 'newly updated', 'updated', 'remodeled', 'new kitchen',
    #         'new bath', 'move-in ready', 'turnkey', 'turn key'
    #     ])
    # }



    # # Keywords that signal a property can achieve a PREMIUM After-Repair Value (ARV)
    # PREMIUM_FINISH_KEYWORDS: dict = {
    #     'has_granite': 'granite',
    #     'has_hardwood': 'hardwood|wood floor',
    #     'has_stainless': 'stainless steel|stainless appliances',
    # }

    # # Keywords that signal a property is in POOR condition, implying HIGH renovation costs.
    # HIGH_COST_KEYWORDS: list = [
    #     'fixer', 'fixer-upper', 'tlc', 'as-is', 'as is', 'rehab',
    #     'investor special', 'needs work', 'handyman', 'opportunity'
    # ]
    
    # # Keywords that signal a property is in GOOD condition, implying LOW renovation costs.
    # LOW_COST_KEYWORDS: list = [
    #     'fully renovated', 'newly updated', 'updated', 'remodeled', 'new kitchen',
    #     'new bath', 'move-in ready', 'turnkey', 'turn key'
    # ]

# Create a single instance of the settings to be imported by other modules.
settings = Settings()