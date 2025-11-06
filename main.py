#!/usr/bin/env python3
"""
🎯 Data Mining Project - Data Cleaning & Understanding Application

Questo script consente di:
- Pulire un dataset (`-c` o `--cleaning`)
- Analizzare il dataset pulito (`-u` o `--understanding`)
- Fare entrambe le operazioni (`-c -u` oppure senza argomenti)
- Opzionalmente generare scatterplot durante l'understanding (`-u -s`)

Esempi:
    python main.py -c
    python main.py -u
    python main.py -u -s
    python main.py -c -u
    python main.py      # esegue tutto
"""

from pathlib import Path
from ast import literal_eval
import argparse
import pandas as pd
from dmp.data_cleaning import clean_df
from dmp.data_understanding import understand_df
from dmp import config

# 🎨 Colori ANSI per una stampa più leggibile
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def clean_data(input_file: Path, output_file: Path):
    """Esegue il data cleaning e salva il risultato."""

    print(f"{Colors.BLUE}📂 Caricamento dataset da:{Colors.RESET} {input_file}")
    df = pd.read_csv(input_file)

    print(f"{Colors.CYAN}🧹 Avvio della pulizia del DataFrame...{Colors.RESET}")
    df_cleaned = clean_df(df)
    print(f"{Colors.GREEN}✨ Pulizia completata con successo!{Colors.RESET}\n")

    print(f"{Colors.YELLOW}💾 Salvataggio del dataset pulito in:{Colors.RESET} {output_file}")
    df_cleaned.to_csv(output_file, index=False)

    return df_cleaned

def understand_data(input_file: Path, do_scatters, do_hists):
    """Esegue le analisi sul dataframe pulito necessarie per svolgere il data understanding."""

    print(f"{Colors.BLUE}📊 Caricamento dataset pulito da:{Colors.RESET} {input_file}")
    df_cleaned = pd.read_csv(input_file, converters={"Ranks": literal_eval}) #Leggi direttamente la colonna "Ranks" come python list e non stringa

    print(f"{Colors.CYAN}🔍 Avvio dell'analisi per data understanding {Colors.RESET}")
    understand_df(df_cleaned, do_scatters, do_hists)
    print(f"{Colors.GREEN}📈 Analisi completata!{Colors.RESET}\n")

def main():
    """Funzione di accesso principale del programma di Data Mining"""

    parser = argparse.ArgumentParser(
        description="Data Cleaning & Understanding Pipeline"
    )
    parser.add_argument(
        "-c", "--cleaning",
        action="store_true",
        help="Esegui solo la fase di data cleaning"
    )
    parser.add_argument(
        "-u", "--understanding",
        action="store_true",
        help="Esegui solo la fase di data understanding"
    )
    parser.add_argument(
        "-s", "--scatters",
        action="store_true",
        help="(Opzionale) Genera gli scatterplot durante l'understanding"
    )
    parser.add_argument(
        "-hi", "--hists",
        action="store_true",
        help="(Opzionale) Genera gli istogrammi durante l'understanding"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Attiva print di log più esaustivi"
    )

    args = parser.parse_args()

    config.set_verbose(args.verbose)

    # Se non viene specificato nulla → esegui tutto
    if not args.cleaning and not args.understanding:
        args.cleaning = args.understanding = True

    # Percorsi base
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    input_file = data_dir / "DM1_game_dataset.csv"
    output_file = data_dir / "cleaned_df.csv"

    print(f"{Colors.HEADER}{Colors.BOLD}🚀 Data Pipeline Avviata!{Colors.RESET}\n")

    # Esecuzione task
    if args.cleaning:
        _ = clean_data(input_file, output_file)

    if args.understanding:
        if not output_file.exists():
            print(f"{Colors.RED}❌ Errore: il file pulito non esiste. Esegui prima con -c o --cleaning.{Colors.RESET}")
            return
        # Passa il flag --scatters come parametro
        understand_data(output_file, args.scatters, args.hists)

    print(f"{Colors.BOLD}🏁 Operazione completata!{Colors.RESET} ✅")

if __name__ == "__main__":
    main()
