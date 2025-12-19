import pandas as pd
from fim import apriori
from itertools import chain
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

from dmp.utils import save_figure

# --- 1. FUNZIONE PER SALVARE I RISULTATI COMPLETI ---
def save_results_to_txt(df, filepath, mode="itemsets"):
    """
    Salva l'intero DataFrame in un file di testo formattato, bypassando i limiti di visualizzazione.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"--- RISULTATI COMPLETI: {mode.upper()} ---\n")
        f.write(f"Totale elementi trovati: {len(df)}\n")
        f.write("-" * 60 + "\n\n")
        
        if df.empty:
            f.write("Nessun risultato trovato.\n")
            return

        if mode == "rules":
            # Formato specifico per le regole
            for i, row in df.iterrows():
                rule = f"{row['antecedent']} => {row['consequent']}"
                stats = f"[Supp: {row['support']}% | Conf: {row['confidence']}% | Lift: {row['lift']}]"
                f.write(f"{i+1}. {rule}\n")
                f.write(f"    {stats}\n")
                f.write("-" * 40 + "\n")
        else:
            # Formato tabellare per itemsets
            f.write(df.to_string(index=False))

# --- 2. UTILITY DATA CLEANING ---
def ensure_list(val):
    if pd.isna(val): return []
    if isinstance(val, (set, list, tuple)): return list(val)
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('{') and val.endswith('}'):
            try: return list(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                return val.replace('{', '').replace('}', '').replace("'", "").replace('"', '').split(', ')
        return val.replace(',', ' ').split()
    return []

# --- 3. FUNZIONI CORE MODIFICATE ---

def make_apriori_for_itemsets(df, columns, supp, zmin, zmax, target_type='frequent', 
                              use_descriptors=False, min_item_count=2, print_info=True, 
                              output_file=None):
    """
    Esegue Apriori per itemset.
    
    Args:
        output_file (str): Percorso dove salvare TUTTI i risultati trovati (txt).
    """
    
    target_map = {'frequent': 's', 's': 's', 'closed': 'c', 'c': 'c', 'maximal': 'm', 'm': 'm'}
    target_char = target_map.get(str(target_type).lower(), 's')

    # FASE 1: Preparazione
    transactions = []
    if use_descriptors:
        col_name = 'Description' if 'Description' in df.columns else 'descriptors'
        if col_name not in df.columns: raise ValueError(f"Colonna '{col_name}' non trovata.")
        transactions = df[col_name].apply(ensure_list).tolist()
    else:
        data = df[columns].copy()
        transactions = data.apply(lambda row: [f"{col}={str(row[col])}" for col in columns], axis=1).tolist()

    # FASE 2: Filtraggio
    if min_item_count > 1:
        all_items_initial = list(chain.from_iterable(transactions))
        item_counts = Counter(all_items_initial)
        valid_items = {item for item, count in item_counts.items() if count >= min_item_count}
        transactions = [[item for item in t if item in valid_items] for t in transactions]
        transactions = [t for t in transactions if t]

    # FASE 3: Report Dataset (Solo a video se richiesto)
    if print_info:
        n_final = len(transactions)
        if n_final == 0:
            print("!!! ATTENZIONE: Dataset vuoto dopo il filtraggio.")
            return pd.DataFrame()
        
        lens = [len(t) for t in transactions]
        avg_len = np.mean(lens) if lens else 0
        
        print(f"\n{'='*20} REPORT DATASET (Itemsets) {'='*20}")
        print(f"Target: {target_type} | Mode: {'Descrittori' if use_descriptors else 'Colonne'}")
        print(f"Transazioni finali: {n_final} | Lunghezza media: {avg_len:.2f}")

    # FASE 4: Esecuzione
    if not transactions: return pd.DataFrame(columns=["itemset", "support"])

    results = apriori(transactions, target=target_char, supp=supp, zmin=zmin, zmax=zmax, report="S")

    res_df = pd.DataFrame(results, columns=["itemset", "support"])
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce').round(2)
    res_df = res_df.sort_values(by="support", ascending=False).reset_index(drop=True)

    # FASE 5: Stampa Anteprima e Salvataggio
    if print_info:
        print(f"\n{'='*20} RISULTATI {'='*20}")
        print(f"Totale Itemset Trovati: {len(res_df)}")
        if not res_df.empty:
            print("--- ANTEPRIMA TOP 5 ---")
            print(res_df.head(5).to_string(index=False))
        print(f"{'='*50}\n")
    
    # SALVATAGGIO COMPLETO SU FILE
    if output_file and not res_df.empty:
        save_results_to_txt(res_df, output_file, mode="itemsets")
        print(f"-> File risultati salvato in: {output_file}")
    elif output_file:
         print("-> Nessun file salvato (nessun risultato).")

    return res_df

def make_apriori_association_rules(df, columns, supp, conf, zmin, zmax, target_char='r', 
                                   use_descriptors=False, min_item_count=2, print_info=True, 
                                   output_file=None):
    """
    Esegue Apriori per Regole.
    
    Args:
        output_file (str): Percorso dove salvare TUTTE le regole trovate (txt).
    """

    # FASE 1 & 2: Preparazione (Uguale a sopra)
    if use_descriptors:
        col_name = 'Description' if 'Description' in df.columns else 'descriptors'
        transactions = df[col_name].apply(ensure_list).tolist()
    else:
        transactions = df[columns].apply(lambda row: [f"{col}={str(row[col])}" for col in columns], axis=1).tolist()

    if min_item_count > 1:
        all_items = list(chain.from_iterable(transactions))
        item_counts = Counter(all_items)
        valid_items = {item for item, count in item_counts.items() if count >= min_item_count}
        transactions = [[item for item in t if item in valid_items] for t in transactions]
        transactions = [t for t in transactions if t] 

    # FASE 3: Esecuzione
    results = apriori(transactions, target='r', supp=supp, conf=conf, zmin=zmin, zmax=zmax, report="SCl")
    
    if not results:
        if print_info: print("Nessuna regola trovata.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results, columns=["consequent", "antecedent", "support", "confidence", "lift"])
    res_df = res_df[["antecedent", "consequent", "support", "confidence", "lift"]] # Riordino
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce').round(2)
    res_df['confidence'] = pd.to_numeric(res_df['confidence'], errors='coerce').round(2)
    res_df['lift'] = pd.to_numeric(res_df['lift'], errors='coerce').round(3)
    res_df = res_df.sort_values(by="lift", ascending=False).reset_index(drop=True)

    # FASE 5: Stampa Anteprima e Salvataggio
    if print_info:
        print(f"\n{'='*20} REGOLE TROVATE {'='*20}")
        print(f"Totale Regole: {len(res_df)}")
        if not res_df.empty:
            print("--- ANTEPRIMA TOP 5 (per Lift) ---")
            temp_df = res_df.head(5).copy()
            for i, row in temp_df.iterrows():
                print(f"{row['antecedent']} => {row['consequent']} [L: {row['lift']}]")
        print(f"{'='*50}\n")

    # SALVATAGGIO COMPLETO SU FILE
    if output_file and not res_df.empty:
        save_results_to_txt(res_df, output_file, mode="rules")
        print(f"-> File risultati salvato in: {output_file}")

    return res_df

def analyze_apriori_sensitivity(df, columns, target_col=None, zmin=2, zmax=5, 
                                use_descriptors=False, min_item_count=2):
    """
    Genera grafici di sensibilità. 
    """
    # Definisce cartella output standard se serve (per ora non usata qui dentro se non per i plot)
    output_folder = "figures/pattern_mining"

    mode_suffix = "descriptors" if use_descriptors else "columns"
    support_range = list(range(1, 25, 2))
    counts_type = {'closed': [], 'maximal': []}
    
    print(f"--- Generazione Sensitivity Plot ({mode_suffix}) ---")

    for supp in support_range:
        res_c = make_apriori_for_itemsets(df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='closed', 
                                          use_descriptors=use_descriptors, min_item_count=min_item_count, print_info=False)
        counts_type['closed'].append(len(res_c))
        
        res_m = make_apriori_for_itemsets(df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='maximal', 
                                          use_descriptors=use_descriptors, min_item_count=min_item_count, print_info=False)
        counts_type['maximal'].append(len(res_m))

    # --- PLOT 1 ---
    plt.figure(figsize=(8, 6))
    plt.plot(support_range, counts_type['maximal'], label='maximal', linewidth=2, marker='o')
    plt.plot(support_range, counts_type['closed'], label='closed', linewidth=2, marker='s')
    plt.xlabel('%support')
    
    if use_descriptors:
        plt.yscale('log')
        plt.ylabel('itemsets (log scale)')
        plt.title(f'Sensitivity: Maximal vs Closed ({mode_suffix}) - Log Scale')
    else:
        plt.ylabel('itemsets')
        plt.title(f'Sensitivity: Maximal vs Closed ({mode_suffix})')

    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    save_figure(plt, title=f"sensitivity_closed_vs_maximal_{mode_suffix}", folder=output_folder)
    print("-> Plot 1 salvato.")

    # --- PLOT 2 (Se c'è target_col) ---
    if target_col and target_col in df.columns:
        print(f"--- Generazione Plot Classi su '{target_col}' ---")
        
        classes = df[target_col].dropna().unique()
        if len(classes) > 5:
            classes = df[target_col].value_counts().head(5).index.tolist()
            
        counts_class = {cls: [] for cls in classes}
        
        for supp in support_range:
            for cls in classes:
                sub_df = df[df[target_col] == cls].copy()
                curr_min_item = 100 if use_descriptors else min_item_count
                
                res = make_apriori_for_itemsets(sub_df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='closed', 
                                                use_descriptors=use_descriptors, min_item_count=curr_min_item, print_info=False)
                counts_class[cls].append(len(res))

        plt.figure(figsize=(8, 6))
        for cls in classes:
            plt.plot(support_range, counts_class[cls], label=f'{target_col}={cls}', linewidth=2, marker='.')
            
        plt.xlabel('%support')
        if use_descriptors:
            plt.yscale('log')
            plt.ylabel('itemsets (log scale)')
            plt.title(f'Itemsets count by Class ({target_col}) - {mode_suffix} - Log Scale')
        else:
            plt.ylabel('itemsets')
            plt.title(f'Itemsets count by Class ({target_col}) - {mode_suffix}')
            
        plt.legend()
        plt.grid(True, alpha=0.3, which="both")
        
        save_figure(plt, title=f"sensitivity_by_{target_col}_{mode_suffix}", folder=output_folder)
        print("-> Plot 2 salvato.")