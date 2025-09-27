# generate_llm_features.py

import pandas as pd
import argparse
import os
from tqdm import tqdm
from config import settings
from scoring_logic import analyze_description_with_llm # Re-use the API's logic

# --- Configuration for the script ---

DATASET_CONFIG = {
    "sold": {
        "input_path": settings.SOLD_PROPERTIES_CSV,
        "output_path": settings.ENRICHED_SOLD_PROPERTIES_CSV
    },
    "for_sale": {
        "input_path": settings.FOR_SALE_PROPERTIES_CSV,
        "output_path": settings.ENRICHED_FOR_SALE_PROPERTIES_CSV
    }
}
BATCH_SIZE = 50 # Process and save in chunks of 50 to prevent data loss on error

def process_in_batches(input_filepath: str, output_filepath: str):
    """
    Reads an input CSV, processes descriptions in batches, and appends the
    enriched data to an output CSV, allowing for resumption.
    """
    print(f"--- Starting Batch Processing ---")
    print(f"Input file: {input_filepath}")
    print(f"Output file: {output_filepath}")

    # --- 1. Resumption Logic: Find Already Processed IDs ---
    # make the script fail-safe and resumable.
    # check the output file first. If it exists, start from the unprocessed ones.
    processed_ids = set()
    if os.path.exists(output_filepath):
        try:
            df_existing = pd.read_csv(output_filepath)
            processed_ids = set(df_existing['property_id'].unique())
            print(f"Found {len(processed_ids)} properties already processed. They will be skipped.")
        except pd.errors.EmptyDataError:
            print("Output file exists but is empty. Starting from scratch.")
        except Exception as e:
            print(f"Could not read existing output file. Error: {e}. Starting from scratch.")

    # --- 2. Load and Filter New Data ---
    df_raw = pd.read_csv(input_filepath)
    df_to_process = df_raw[~df_raw['property_id'].isin(processed_ids)].copy()
    df_to_process['text'] = df_to_process['text'].fillna('')

    if df_to_process.empty:
        print("No new properties to process. All data is up to date.")
        return

    print(f"Found {len(df_to_process)} new properties to analyze.")
    
    # --- 3. Batch Processing Loop ---
    for i in tqdm(range(0, len(df_to_process), BATCH_SIZE), desc="Overall Progress"):
        df_batch = df_to_process.iloc[i:i + BATCH_SIZE]
        
        tqdm.pandas(desc=f"Processing Batch {i//BATCH_SIZE + 1}")
        
        # Apply the LLM analysis to the current batch
        llm_results = df_batch['text'].progress_apply(analyze_description_with_llm)
        
        # Assemble the results for this batch
        llm_df = pd.DataFrame([res.model_dump() for res in llm_results])
        llm_df.rename(columns={
            'quality_score': 'llm_quality_score',
            'risk_score': 'llm_risk_score'
        }, inplace=True)
        
        # Combine with original data (excluding the now redundant 'text' column)
        batch_enriched = pd.concat([
            df_batch.drop(columns=['text']).reset_index(drop=True), 
            llm_df.reset_index(drop=True)
        ], axis=1)

        # --- 4. Fail-Safe Save ---
        is_new_file = not os.path.exists(output_filepath)
        batch_enriched.to_csv(output_filepath, mode='a', header=is_new_file, index=False)
        
        tqdm.write(f"Batch {i//BATCH_SIZE + 1} completed and saved.")

    print("\n--- All batches processed and saved successfully! ---")

def main():
    """Main function to parse arguments and orchestrate the process."""
    parser = argparse.ArgumentParser(description="Generate LLM features for real estate data.")
    parser.add_argument(
        "dataset_type",
        choices=["sold", "for_sale"],
        help="The type of dataset to process: 'sold' (for training) or 'for_sale' (for inference)."
    )
    args = parser.parse_args()

    config = DATASET_CONFIG[args.dataset_type]
    process_in_batches(config["input_path"], config["output_path"])

if __name__ == "__main__":
    main()