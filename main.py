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
from yaspin import yaspin
from dmp.utils import filter_columns
from dmp.data_cleaning import clean_df
from dmp.data_understanding import understand_df
from dmp.data_preparation import prepare_df
from dmp.data_clustering import cluster_df
from dmp.config import VERBOSE, set_verbose

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

def clean_data(input_file: Path, cleaned_output_file: Path, filtered_output_file: Path, verbose: bool):
    """Esegue il data cleaning e salva il risultato."""

    print(f"{Colors.BLUE}📂 Caricamento dataset da:{Colors.RESET} {input_file}")
    df = pd.read_csv(input_file)

    if verbose:
        print(f"{Colors.CYAN}🧹 Avvio della pulizia del DataFrame...{Colors.RESET}")
        df_cleaned = clean_df(df)
    else:
        with yaspin(text="🧹 Avvio della pulizia del DataFrame...", color="cyan") as spinner:
            df_cleaned = clean_df(df)
            spinner.ok("✅")

    print(f"{Colors.GREEN}✨ Pulizia completata con successo!{Colors.RESET}\n")

    columns=[
    "YearPublished", "GameWeight", "ComWeight",  
    "ComAgeRec", "LanguageEase", "NumOwned", "NumWant", "NumWish","MfgPlaytime",
    "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "NumUserRatings",
    ]
    
    # Definisco il df filtrato dagli outliers, posso chiamarla più volte per filtrare in modo modulare il df e lo salva.
    df_filtered = filter_columns(df_cleaned, colonne=columns, method="percentile", params=None, delete_row=False)

    print(f"{Colors.YELLOW}💾 Salvataggio del dataset pulito in:{Colors.RESET} {cleaned_output_file}")
    df_cleaned.to_csv(cleaned_output_file, index=False)
    print(f"{Colors.YELLOW}💾 Salvataggio del dataset pulito in:{Colors.RESET} {filtered_output_file}")
    df_filtered.to_csv(filtered_output_file, index=False)

def understand_data(cleaned_file: Path, filtered_file:Path, do_scatters, do_hists, descriptors, verbose: bool):
    """Esegue le analisi sul dataframe pulito necessarie per svolgere il data understanding."""

    print(f"{Colors.BLUE}📊 Caricamento dataset pulito da:{Colors.RESET} {cleaned_file}")
    df_cleaned = pd.read_csv(cleaned_file, converters={"Ranks": literal_eval}) #Leggi direttamente la colonna "Ranks" come python list e non stringa

    print(f"{Colors.BLUE}📊 Caricamento dataset filtrato da:{Colors.RESET} {filtered_file}")
    df_filtered = pd.read_csv(filtered_file, converters={"Ranks": literal_eval})

    if verbose:
        print(f"{Colors.CYAN}🔍 Avvio dell'analisi per data understanding {Colors.RESET}")
        understand_df(df_cleaned, df_filtered, do_scatters, do_hists, descriptors)
    else:
        with yaspin(text="🔍 Avvio dell'analisi per data understanding ", color="cyan") as spinner:
            understand_df(df_cleaned, df_filtered, do_scatters, do_hists, descriptors)
            spinner.ok("✅")

    print(f"{Colors.GREEN}📈 Analisi completata!{Colors.RESET}\n")

def prepare_data(input_file: Path, output_file: Path, N_samples, descriptors, verbose: bool):
    """Esegue la preparazione del dataframe, prendendo un dataframe pulito e 
    riducendo ulteriormente la sua dimensione"""
    
    print(f"{Colors.BLUE}📊 Caricamento dataset pulito da:{Colors.RESET} {input_file}")
    filtered_df = pd.read_csv(input_file, converters={"Ranks": literal_eval}) #Leggi direttamente la colonna "Ranks" come python list e non stringa

    if verbose:
        print(f"{Colors.CYAN}🔍 Avvio della preparazione dei dati {Colors.RESET}")
        prepared_df = prepare_df(filtered_df, N_samples, descriptors, hists=True)
    else:
        with yaspin(text="🔍 Avvio della preparazione dei dati ", color="cyan") as spinner:
            prepared_df = prepare_df(filtered_df, N_samples, descriptors, hists=True)
            spinner.ok("✅")
    print(f"{Colors.GREEN}📈 Preparazione completata!{Colors.RESET}\n")  
    
    print(f"{Colors.YELLOW}💾 Salvataggio del dataset filtrato in:{Colors.RESET} {output_file}")
    prepared_df.to_csv(output_file, index=False)

def cluster_data(input_file: Path, verbose: bool):
    """Esegue la clusterizzazione di alcune colonne selezionate sul dataframe già pulite e filtrato"""

    print(f"{Colors.BLUE}📊 Caricamento dataset pulito da:{Colors.RESET} {input_file}")
    filtered_df = pd.read_csv(input_file, converters={"Ranks": literal_eval}) #Leggi direttamente la colonna "Ranks" come python list e non stringa

    if verbose:
        print(f"{Colors.CYAN}🔍 Avvio della clusterizzazione dei dati {Colors.RESET}")
        cluster_data(filtered_df)
    else:
        with yaspin(text="🔍 Avvio della clusterizzazione dei dati ", color="cyan") as spinner:
            cluster_df(filtered_df)
            spinner.ok("✅")
    print(f"{Colors.GREEN}📈 Clusterizzazione completata!{Colors.RESET}\n")  

def hypno_toad():
    print(r"""
      ,'``.._   ,'``.
     :,--._:)\,:,._,.:       All Glory to
     :`--,''   :`...';\      the HYPNO TOAD!
      `,'       `---'  `.
      /                 :
     /                   \
   ,'                     :\.___,-.
  `...,---'``````-..._    |:       \
    (                 )   ;:    )   \  _,-.
     `.              (   //          `'    \
      :               `.//  )      )     , ;
    ,-|`.            _,'/       )    ) ,' ,'
   (  :`.`-..____..=:.-':     .     _,' ,'
    `,'\ ``--....-)='    `._,  \  ,') _ '``._
 _.-/ _ `.       (_)      /     )' ; / \ \`-.'
`--(   `-:`.     `' ___..'  _,-'   |/   `.)
    `-. `.`.``-----``--,  .'
      |/`.\`'        ,','); SSt
          `         (/  (/
    """)

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
        "-p", "--preparation",
        nargs="?",            # rende l'argomento opzionale
        const=None,           # valore usato se -p è presente ma senza numero
        type=int,             # se viene passato un numero, viene convertito
        metavar="N",
        help="Esegui la fase di data preparation (opzionale: specifica N per campionare le righe)"
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
    parser.add_argument(
        "-hypno", "--hypnotoad",
        action="store_true",
        help="GLORY TO THE HYPNO TOAD"
    )
    parser.add_argument(
        "-d", "--descriptors",
        nargs="+",                # accetta uno o più valori
        help="Filtra il dataset per le righe che contengono uno o più descrittori nella colonna 'Description'"
    )
    parser.add_argument(
        "-cl", "--clustering",
        action="store_true",
        help="Esegue il clutering di alcune colonne del DataFrame"
    )

    args = parser.parse_args()

    set_verbose(args.verbose)

    # Se non viene specificato nessuno dei tre step → esegui tutto
    if not args.cleaning and not args.understanding and not args.clustering:
        args.cleaning = args.understanding = True

    # Percorsi base
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    input_file = data_dir / "DM1_game_dataset.csv"
    cleaned_output_file = data_dir / "cleaned_df.csv" # Percorso per dataset pulito
    filtered_output_file = data_dir / "filtered_df.csv" # Percorso per dataset con rimozione outliers
    prepared_output_file = data_dir / "prepared_df.csv" # Percorso per dataset con data prep

    print(f"{Colors.HEADER}{Colors.BOLD}🚀 Data Pipeline Avviata!{Colors.RESET}\n")

    # Esecuzione task
    if args.cleaning:
        clean_data(input_file, cleaned_output_file, filtered_output_file, args.verbose)
        if args.preparation is None or isinstance(args.preparation, int):
            prepare_data(filtered_output_file, prepared_output_file, args.preparation, args.descriptors, args.verbose)



    if args.understanding:
        if not cleaned_output_file.exists() and filtered_output_file.exists():
            print(f"{Colors.RED}❌ Errore: il file pulito non esiste. Esegui prima con -c o --cleaning.{Colors.RESET}")
            return
        understand_data(cleaned_output_file, filtered_output_file, args.scatters, args.hists, args.descriptors, args.verbose)


    if args.clustering:
        cluster_data(prepared_output_file, args.verbose)

    print(f"{Colors.BOLD}🏁 Operazione completata!{Colors.RESET} ✅")

    if args.hypnotoad:
        hypno_toad()

if __name__ == "__main__":
    main()

