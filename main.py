#!/usr/bin/env python3
"""
Data Mining Project - Data Cleaning Application
Questo script carica un dataset, lo pulisce utilizzando la funzione `clean_df`
"""

import pandas as pd
from pathlib import Path
from dmp import clean_df

def main():
    # Definisci i percorsi
    data_dir = Path("data")
    input_file = data_dir / "DM1_game_dataset.csv"
    output_file = data_dir / "cleaned_df.csv"

    # Assicurati che la directory dei dati esista
    data_dir.mkdir(exist_ok=True)
    
    # Carica il dataset
    print(f"Caricando dataset da {input_file}")
    df = pd.read_csv(input_file)
    print(f"Dataset caricato con successo con {len(df)} righe.")

    # Pulisci il dataframe
    df_cleaned = clean_df(df)

    # Salva il dataset pulito
    print(f"Salvando il dataset come {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print("Salvato con successo.")
    
if __name__ == "__main__":
    main()