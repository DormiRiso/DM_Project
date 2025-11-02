#!/usr/bin/env python3
"""
Data Mining Project - Data Cleaning Application
This script loads the game dataset and performs data cleaning operations.
"""

import pandas as pd
from pathlib import Path
from dmp import clean_df

def main():
    # Define paths
    data_dir = Path("data")
    input_file = data_dir / "DM1_game_dataset.csv"
    output_file = data_dir / "cleaned_df.csv"

    # Ensure data directory exists
    data_dir.mkdir(exist_ok=True)

    print("\n=== Starting Data Cleaning Process ===\n")
    
    # Load the dataset
    print(f"Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    print(f"Successfully loaded dataset with {len(df)} rows.")

    # Clean the dataframe
    print("\nCleaning the dataset...")
    df_cleaned = clean_df(df)
    print("Dataset cleaning completed.")

    # Save the cleaned dataset
    print(f"\nSaving cleaned dataset to {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print("Cleaned dataset saved successfully.")
    
    print("\n=== Data Cleaning Process Completed ===\n")

if __name__ == "__main__":
    main()