#!/usr/bin/env python3
"""
🎯 Data Mining Project - Data Cleaning Application
Questo script carica un dataset, lo pulisce utilizzando la funzione `clean_df`
"""

import pandas as pd
from pathlib import Path
from dmp.data_cleaning import clean_df
from dmp.data_understanding import understand_df

# Colori ANSI per una stampa più leggibile
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# Colori ANSI per una stampa più leggibile
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 Data Cleaning Pipeline Avviata!{Colors.RESET}\n")

    # Definisci i percorsi
    data_dir = Path("data")
    input_file = data_dir / "DM1_game_dataset.csv"
    output_file = data_dir / "cleaned_df.csv"

    # Assicurati che la directory dei dati esista
    data_dir.mkdir(exist_ok=True)

    # Carica il dataset
    print(f"{Colors.BLUE}📂 Caricamento dataset da:{Colors.RESET} {input_file}")
    df = pd.read_csv(input_file)
    print(f"{Colors.GREEN}✅ Dataset caricato con successo!{Colors.RESET}")
    print(f"   📊 Righe totali: {Colors.BOLD}{len(df)}{Colors.RESET}\n")

    # Pulisci il dataframe
    print(f"{Colors.CYAN}🧹 Avvio della pulizia del DataFrame...{Colors.RESET}")
    df_cleaned = clean_df(df)
    print(f"{Colors.GREEN}✨ Pulizia completata con successo!{Colors.RESET}\n")

    # Salva il dataset pulito
    print(f"{Colors.YELLOW}💾 Salvataggio del dataset pulito in:{Colors.RESET} {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print(f"{Colors.GREEN}🎉 File salvato correttamente!{Colors.RESET}\n")

    print(f"{Colors.BOLD}🏁 Operazione completata!{Colors.RESET} ✅")
        
    ###############################################


    #Fai data understanding sul df pulito
    understand_df(df_cleaned)

if __name__ == "__main__":
    main()
