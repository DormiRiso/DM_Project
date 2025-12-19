import pandas as pd
from fim import apriori, fpgrowth
from itertools import chain
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

from dmp.utils import save_figure

# =============================================================================
# 1. UTILITIES DI SUPPORTO
# =============================================================================

def save_results_to_txt(df, filepath, mode="itemsets"):
    """
    Salva il DataFrame dei risultati in un file .txt formattato.
    Bypassa i limiti di visualizzazione della console.
    """
    # Assicura che la cartella esista
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"--- RISULTATI COMPLETI: {mode.upper()} ---\n")
        f.write(f"Totale elementi trovati: {len(df)}\n")
        f.write("-" * 60 + "\n\n")
        
        if df.empty:
            f.write("Nessun risultato trovato.\n")
            return

        if mode == "rules":
            # Formato specifico leggibile per le regole
            for i, row in df.iterrows():
                rule = f"{row['antecedent']} => {row['consequent']}"
                stats = f"[Supp: {row['support']}% | Conf: {row['confidence']}% | Lift: {row['lift']}]"
                f.write(f"{i+1}. {rule}\n")
                f.write(f"    {stats}\n")
                f.write("-" * 40 + "\n")
        else:
            # Formato tabellare standard per gli itemset
            f.write(df.to_string(index=False))

def ensure_list(val):
    """
    Converte stringhe, set o tuple in una lista pulita.
    Gestisce casi di parsing da CSV (es. stringhe che sembrano liste).
    """
    if pd.isna(val): return []
    if isinstance(val, (set, list, tuple)): return list(val)
    if isinstance(val, str):
        val = val.strip()
        # Gestione stringa che rappresenta un set/list: "{'a', 'b'}"
        if val.startswith('{') and val.endswith('}'):
            try: return list(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                # Fallback manuale se ast fallisce
                return val.replace('{', '').replace('}', '').replace("'", "").replace('"', '').split(', ')
        # Gestione stringa semplice separata da virgole
        return val.replace(',', ' ').split()
    return []

# =============================================================================
# 2. FUNZIONI CORE DI MINING
# =============================================================================

def do_pattern_mining_for_itemsets(df, columns, supp, zmin, zmax, target_type='frequent', 
                              use_descriptors=False, min_item_count=2, print_info=True, 
                              output_file=None, algo='apriori'):
    """
    Esegue il Pattern Mining (usando Apriori o FP-Growth) per estrarre itemset.

    Args (Input):
        df (pd.DataFrame): Il DataFrame contenente i dati.
        columns (list): Lista dei nomi delle colonne da analizzare (usato se use_descriptors=False).
        supp (float): Supporto minimo in percentuale (es. 10 per 10%). Frequenza minima di apparizione.
        zmin (int): Numero minimo di elementi in un itemset (es. 2).
        zmax (int): Numero massimo di elementi in un itemset (es. 5). Cruciale per evitare blocchi.
        target_type (str, optional): Tipo di pattern. Default 'frequent'.
            - 'frequent' ('s'): Tutti gli itemset frequenti.
            - 'closed' ('c'): Itemset chiusi (nessun superset ha lo stesso supporto).
            - 'maximal' ('m'): Itemset massimali (nessun superset è frequente).
        use_descriptors (bool, optional): Se True, analizza la colonna 'Description' (liste di parole). Default False.
        min_item_count (int, optional): Filtra elementi che appaiono meno di N volte totali nel dataset. Default 2.
        print_info (bool, optional): Se True, stampa report a video. Default True.
        output_file (str, optional): Percorso file .txt dove salvare i risultati completi. Default None.
        algo (str, optional): Algoritmo da usare: 'apriori' o 'fpgrowth'. Default 'apriori'.

    Returns (Output):
        pd.DataFrame: Un DataFrame con due colonne:
            - 'itemset': La tupla degli elementi trovati.
            - 'support': La percentuale di supporto.
            Restituisce un DataFrame vuoto se non trova nulla.
    """
    
    # Mappatura target per pyfim
    target_map = {'frequent': 's', 's': 's', 'closed': 'c', 'c': 'c', 'maximal': 'm', 'm': 'm'}
    target_char = target_map.get(str(target_type).lower(), 's')

    # --- FASE 1: Preparazione Transazioni ---
    transactions = []
    if use_descriptors:
        col_name = 'Description' if 'Description' in df.columns else 'descriptors'
        if col_name not in df.columns: raise ValueError(f"Colonna '{col_name}' non trovata.")
        transactions = df[col_name].apply(ensure_list).tolist()
    else:
        data = df[columns].copy()
        transactions = data.apply(lambda row: [f"{col}={str(row[col])}" for col in columns], axis=1).tolist()

    # --- FASE 2: Filtraggio Item Rari (Pruning) ---
    if min_item_count > 1:
        all_items_initial = list(chain.from_iterable(transactions))
        item_counts = Counter(all_items_initial)
        valid_items = {item for item, count in item_counts.items() if count >= min_item_count}
        transactions = [[item for item in t if item in valid_items] for t in transactions]
        transactions = [t for t in transactions if t]

    # --- FASE 3: Report Dataset ---
    if print_info:
        n_final = len(transactions)
        if n_final == 0:
            print(f"[{algo.upper()}] ATTENZIONE: Dataset vuoto dopo il filtraggio.")
            return pd.DataFrame()
        
        lens = [len(t) for t in transactions]
        avg_len = np.mean(lens) if lens else 0
        
        print(f"\n--- REPORT ITEMSETS ({algo.upper()}) ---")
        print(f" Target: {target_type} | Mode: {'Descrittori' if use_descriptors else 'Colonne'}")
        print(f" Transazioni: {n_final} | Len Media: {avg_len:.2f}")

    # --- FASE 4: Esecuzione Algoritmo ---
    if not transactions: return pd.DataFrame(columns=["itemset", "support"])

    if algo.lower() == 'fpgrowth':
        results = fpgrowth(transactions, target=target_char, supp=supp, zmin=zmin, zmax=zmax, report="S")
    else:
        results = apriori(transactions, target=target_char, supp=supp, zmin=zmin, zmax=zmax, report="S")

    res_df = pd.DataFrame(results, columns=["itemset", "support"])
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce').round(2)
    res_df = res_df.sort_values(by="support", ascending=False).reset_index(drop=True)

    # --- FASE 5: Output ---
    if print_info:
        print(f" > Itemset trovati: {len(res_df)}")
        if not res_df.empty:
            print(" > Top 5 Itemset:")
            print(res_df.head(5).to_string(index=False))
        print("-" * 40)
    
    if output_file and not res_df.empty:
        save_results_to_txt(res_df, output_file, mode="itemsets")
        if print_info: print(f" > File salvato: {output_file}")
    elif output_file:
         if print_info: print(" > Nessun risultato da salvare.")

    return res_df

def find_association_rules(df, columns, supp, conf, zmin, zmax, target_char='r', 
                                   use_descriptors=False, min_item_count=2, print_info=True, 
                                   output_file=None, algo='apriori'):
    """
    Esegue il Mining per trovare Regole di Associazione (Antecedente -> Conseguente).

    Args (Input):
        df (pd.DataFrame): Il DataFrame contenente i dati.
        columns (list): Lista colonne da usare (se use_descriptors=False).
        supp (float): Supporto minimo % (frequenza della regola nel dataset).
        conf (float): Confidenza minima % (probabilità che B accada dato A).
        zmin (int): Dimensione minima della regola (Antecedente + Conseguente).
        zmax (int): Dimensione massima della regola.
        target_char (str, optional): Parametro interno per pyfim ('r' per rules). Non modificare.
        use_descriptors (bool, optional): Se True, usa la colonna 'Description'. Default False.
        min_item_count (int, optional): Rimuove elementi rari prima dell'analisi. Default 2.
        print_info (bool, optional): Se True, stampa report a video. Default True.
        output_file (str, optional): Percorso file .txt dove salvare tutte le regole. Default None.
        algo (str, optional): Algoritmo da usare: 'apriori' o 'fpgrowth'. Default 'apriori'.

    Returns (Output):
        pd.DataFrame: Un DataFrame ordinato per 'lift' con le colonne:
            - 'antecedent': La causa (es. frozenset({'A', 'B'})).
            - 'consequent': L'effetto (es. frozenset({'C'})).
            - 'support': % presenza combinata.
            - 'confidence': % affidabilità.
            - 'lift': Forza della correlazione (>1 positiva, <1 negativa).
    """
    # --- FASE 1 & 2: Preparazione (Uguale a sopra) ---
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

    # --- FASE 3: Esecuzione Algoritmo ---
    # Report="SCl" -> Support, Confidence, Lift
    if algo.lower() == 'fpgrowth':
        results = fpgrowth(transactions, target='r', supp=supp, conf=conf, zmin=zmin, zmax=zmax, report="SCl")
    else:
        results = apriori(transactions, target='r', supp=supp, conf=conf, zmin=zmin, zmax=zmax, report="SCl")
    
    if not results:
        if print_info: print(f"[{algo.upper()}] Nessuna regola trovata.")
        return pd.DataFrame()

    # Creazione DataFrame
    res_df = pd.DataFrame(results, columns=["consequent", "antecedent", "support", "confidence", "lift"])
    res_df = res_df[["antecedent", "consequent", "support", "confidence", "lift"]] # Riordino colonne
    
    # Arrotondamenti
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce').round(2)
    res_df['confidence'] = pd.to_numeric(res_df['confidence'], errors='coerce').round(2)
    res_df['lift'] = pd.to_numeric(res_df['lift'], errors='coerce').round(3)
    
    # Ordinamento per Lift
    res_df = res_df.sort_values(by="lift", ascending=False).reset_index(drop=True)

    # --- FASE 5: Output ---
    if print_info:
        print(f"\n--- REPORT REGOLE ({algo.upper()}) ---")
        print(f" > Regole trovate: {len(res_df)}")
        if not res_df.empty:
            print(" > Top 5 Regole (per Lift):")
            temp_df = res_df.head(5).copy()
            for i, row in temp_df.iterrows():
                print(f"   {row['antecedent']} => {row['consequent']} [L: {row['lift']}]")
        print("-" * 40)

    if output_file and not res_df.empty:
        save_results_to_txt(res_df, output_file, mode="rules")
        if print_info: print(f" > File salvato: {output_file}")

    return res_df

# =============================================================================
# 3. ANALISI DI SENSIBILITÀ (PLOT)
# =============================================================================

def analyze_sensitivity(df, columns, target_col=None, zmin=2, zmax=5, 
                                use_descriptors=False, min_item_count=2, algo='apriori'):
    """
    Genera e salva grafici di sensibilità variando il supporto.
    Esegue l'algoritmo multiple volte su un range di supporti (da 1% a 25%).

    Args (Input):
        df (pd.DataFrame): Il DataFrame contenente i dati.
        columns (list): Lista colonne da usare.
        target_col (str, optional): Colonna categorica per il confronto tra classi (es. 'WeightedRating').
                                    Se fornita, genera il secondo grafico (Class Comparison).
        zmin (int): Lunghezza minima itemset per l'analisi.
        zmax (int): Lunghezza massima itemset.
        use_descriptors (bool, optional): Se True, usa la colonna 'Description' e attiva scala logaritmica nei plot.
        min_item_count (int, optional): Filtro elementi rari (importante per velocità nei loop).
        algo (str, optional): Algoritmo da usare. Consigliato 'fpgrowth' per velocità. Default 'apriori'.

    Returns (Output):
        None: La funzione non restituisce dati, ma salva i file .png nella cartella 'figures/pattern_mining/{algo}'.
    """
    
    # Definisce la sottocartella specifica per l'algoritmo corrente
    output_folder = f"figures/pattern_mining/{algo}"
    os.makedirs(output_folder, exist_ok=True)

    mode_suffix = "descriptors" if use_descriptors else "columns"
    support_range = list(range(1, 25, 2)) 
    counts_type = {'closed': [], 'maximal': []}
    
    print(f"\n--- Generazione Sensitivity Plot ({mode_suffix}) con {algo.upper()} ---")

    # --- CICLO 1: CLOSED vs MAXIMAL ---
    for supp in support_range:
        # Calcolo Closed (senza stampare info)
        res_c = do_pattern_mining_for_itemsets(
            df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='closed', 
            use_descriptors=use_descriptors, min_item_count=min_item_count, 
            print_info=False, algo=algo
        )
        counts_type['closed'].append(len(res_c))
        
        # Calcolo Maximal (senza stampare info)
        res_m = do_pattern_mining_for_itemsets(
            df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='maximal', 
            use_descriptors=use_descriptors, min_item_count=min_item_count, 
            print_info=False, algo=algo
        )
        counts_type['maximal'].append(len(res_m))

    # --- CREAZIONE PLOT 1 ---
    plt.figure(figsize=(8, 6))
    plt.plot(support_range, counts_type['maximal'], label='maximal', linewidth=2, marker='o')
    plt.plot(support_range, counts_type['closed'], label='closed', linewidth=2, marker='s')
    plt.xlabel('%support')
    
    if use_descriptors:
        plt.yscale('log')
        plt.ylabel('itemsets (log scale)')
        plt.title(f'Sensitivity: Maximal vs Closed ({mode_suffix}) - {algo} - Log Scale')
    else:
        plt.ylabel('itemsets')
        plt.title(f'Sensitivity: Maximal vs Closed ({mode_suffix}) - {algo}')

    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    # Salvataggio
    save_figure(plt, title=f"sensitivity_closed_vs_maximal_{mode_suffix}", folder=output_folder)
    print(" > Plot 1 (Closed vs Maximal) salvato.")

    # --- CICLO 2: CONFRONTO CLASSI (Se target_col esiste) ---
    if target_col and target_col in df.columns:
        print(f" > Analisi per classe su '{target_col}'...")
        
        classes = df[target_col].dropna().unique()
        # Se ci sono troppe classi, prendiamo solo le top 5
        if len(classes) > 5:
            classes = df[target_col].value_counts().head(5).index.tolist()
            
        counts_class = {cls: [] for cls in classes}
        
        for supp in support_range:
            for cls in classes:
                sub_df = df[df[target_col] == cls].copy()
                
                # Ottimizzazione per descrittori: filtro più aggressivo nei loop
                curr_min_item = 100 if use_descriptors else min_item_count
                
                res = do_pattern_mining_for_itemsets(
                    sub_df, columns, supp=supp, zmin=zmin, zmax=zmax, target_type='closed', 
                    use_descriptors=use_descriptors, min_item_count=curr_min_item, 
                    print_info=False, algo=algo
                )
                counts_class[cls].append(len(res))

        # --- CREAZIONE PLOT 2 ---
        plt.figure(figsize=(8, 6))
        for cls in classes:
            plt.plot(support_range, counts_class[cls], label=f'{target_col}={cls}', linewidth=2, marker='.')
            
        plt.xlabel('%support')
        if use_descriptors:
            plt.yscale('log')
            plt.ylabel('itemsets (log scale)')
            plt.title(f'Itemsets by Class ({target_col}) - {mode_suffix} - Log Scale')
        else:
            plt.ylabel('itemsets')
            plt.title(f'Itemsets by Class ({target_col}) - {mode_suffix}')
            
        plt.legend()
        plt.grid(True, alpha=0.3, which="both")
        
        save_figure(plt, title=f"sensitivity_by_{target_col}_{mode_suffix}", folder=output_folder)
        print(" > Plot 2 (Class Comparison) salvato.")