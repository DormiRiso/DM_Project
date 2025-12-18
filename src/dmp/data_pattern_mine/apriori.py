import pandas as pd
from fim import apriori  # Importiamo l'algoritmo Apriori dalla libreria PyFim (veloce implementazione in C)
from itertools import chain
from collections import Counter
import numpy as np
import ast  # Modulo Abstract Syntax Tree: serve per interpretare stringhe come codice Python (es. "{'a', 'b'}" -> set)

def ensure_list(val):
    """
    Funzione di pulizia e conversione.
    
    Il suo scopo è garantire che qualsiasi cosa arrivi dalla colonna 'Description'
    venga trasformata in una lista pulita di stringhe.
    
    Gestisce vari casi "sporchi" che possono capitare leggendo CSV:
    1. Oggetti Python reali (set, list, tuple).
    2. Stringhe che sembrano set: "{'mela', 'pera'}".
    3. Stringhe semplici separate da spazi o virgole.
    4. Valori nulli (NaN).
    """
    # Se il valore è nullo (NaN), restituisce una lista vuota per non rompere il codice successivo
    if pd.isna(val):
        return []
    
    # Se è già una lista, un set o una tupla, lo convertiamo in lista e siamo a posto
    if isinstance(val, (set, list, tuple)):
        return list(val)
        
    # Se è una stringa, dobbiamo "parsearla" (interpretarla)
    if isinstance(val, str):
        val = val.strip()
        
        # Caso: La stringa è la rappresentazione testuale di un set o lista (es. "{'word1', 'word2'}")
        if val.startswith('{') and val.endswith('}'):
            try:
                # ast.literal_eval tenta di valutare la stringa come codice Python in sicurezza
                return list(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                # Fallback: Se ast fallisce (es. sintassi errata), puliamo manualmente le parentesi e splittiamo
                return val.replace('{', '').replace('}', '').replace("'", "").replace('"', '').split(', ')
        
        # Caso: Stringa normale (es. "gioco divertente veloce") o separata da virgole
        return val.replace(',', ' ').split()
        
    # Se non è nessuno dei tipi sopra, restituisce vuoto
    return []

def make_apriori_for_itemsets(df, columns, supp, zmin, zmax, target_type='frequent', use_descriptors=False, min_item_count=2, print_info=True):
    """
    Esegue l'algoritmo Apriori per estrarre gli itemset frequenti (gruppi di elementi che appaiono spesso insieme).

    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati.
        columns (list): Lista dei nomi delle colonne da analizzare (usato solo se use_descriptors=False).
        supp (float): Supporto minimo in percentuale (es. 10 per 10%). Indica la frequenza minima affinché un itemset venga considerato.
        zmin (int): Numero minimo di elementi in un itemset (es. 2 per cercare almeno coppie).
        zmax (int): Numero massimo di elementi in un itemset. Fondamentale per limitare la complessità computazionale.
        target_type (str, optional): Il tipo di pattern da estrarre. Default 'frequent'.
            - 'frequent' ('s'): Tutti gli itemset frequenti.
            - 'closed' ('c'): Itemset chiusi (nessun superset ha lo stesso supporto).
            - 'maximal' ('m'): Itemset massimali (nessun superset è frequente).
        use_descriptors (bool, optional): Se True, ignora `columns` e analizza la colonna 'Description' (testo). Default False.
        min_item_count (int, optional): Filtro pre-elaborazione. Rimuove gli elementi che compaiono nel dataset meno di N volte totali. Utile per pulire descrittori rari. Default 2.
        print_info (bool, optional): Se True, stampa a video statistiche sul dataset e i risultati principali. Default True.

    Returns:
        pd.DataFrame: Un DataFrame con due colonne:
            - 'itemset': La tupla degli elementi trovati.
            - 'support': La percentuale di supporto.
    """
    
    # Mapping per tradurre le richieste utente ('frequent', 'closed') nei codici brevi richiesti da PyFim ('s', 'c')
    target_map = {
        'frequent': 's', 's': 's',  # Frequent (Standard)
        'closed': 'c',   'c': 'c',  # Closed (Nessun superset con stesso supporto)
        'maximal': 'm',  'm': 'm',  # Maximal (Nessun superset frequente)
    }
    # Ottiene il carattere target corretto, default a 's'
    target_char = target_map.get(str(target_type).lower(), 's')

    # --- FASE 1: Preparazione Dati (Costruzione Transazioni) ---
    transactions = []
    if use_descriptors:
        # CASO 1: Analisi del testo (Descrittori)
        # Cerchiamo la colonna corretta
        col_name = 'Description' if 'Description' in df.columns else 'descriptors'
        if col_name not in df.columns:
            raise ValueError(f"Colonna '{col_name}' non trovata.")
            
        # Applichiamo la funzione di pulizia 'ensure_list' a ogni riga per ottenere liste di parole
        transactions = df[col_name].apply(ensure_list).tolist()
    else:
        # CASO 2: Analisi delle colonne tabellari
        # Trasformiamo ogni riga in una lista di stringhe "Colonna=Valore"
        data = df[columns].copy()
        transactions = data.apply(
            lambda row: [f"{col}={str(row[col])}" for col in columns], 
            axis=1
        ).tolist()

    # --- FASE 2: Filtraggio (Pruning) degli item rari ---
    # Questo passaggio è cruciale per la performance: rimuove parole/valori che appaiono pochissimo
    if min_item_count > 1:
        # 1. Mettiamo tutte le parole in un unico calderone per contarle
        all_items_initial = list(chain.from_iterable(transactions))
        item_counts = Counter(all_items_initial)
        
        # 2. Creiamo un set delle parole "valide" (frequenza >= min_item_count)
        valid_items = {item for item, count in item_counts.items() if count >= min_item_count}
        
        # 3. Ricostruiamo le transazioni tenendo solo le parole valide
        cleaned_transactions = []
        for t in transactions:
            # Filtra la lista t mantenendo solo elementi in valid_items
            filtered_t = [item for item in t if item in valid_items]
            # Se la transazione non è vuota dopo il filtro, la teniamo
            if filtered_t:
                cleaned_transactions.append(filtered_t)
        transactions = cleaned_transactions

    # --- FASE 3: Report Informativo (Opzionale) ---
    # Stampa statistiche utili per capire se i parametri sono giusti
    if print_info:
        n_trans_final = len(transactions)
        if n_trans_final == 0:
            print("!!! ATTENZIONE: Dataset vuoto dopo il filtraggio.")
            return pd.DataFrame()

        lengths = [len(t) for t in transactions]
        avg_len = np.mean(lengths) if lengths else 0
        max_len = np.max(lengths) if lengths else 0
        
        print(f"\n{'='*20} REPORT DATASET {'='*20}")
        print(f"Transazioni finali: {n_trans_final}")
        print(f"Lunghezza Transazioni:  Max={max_len}, Media={avg_len:.2f}")
        # Avviso se zmin è troppo alto rispetto ai dati reali
        if zmin > max_len:
            print(f"!!! WARNING: zmin ({zmin}) > lung_max ({max_len}). Nessun risultato possibile.")
        print(f"{'='*56}")

    # --- FASE 4: Esecuzione Apriori (Il cuore dell'algoritmo) ---
    if not transactions:
        return pd.DataFrame(columns=["itemset", "support"])

    # Chiamata alla libreria fim.apriori
    # zmax: Limita la dimensione massima degli itemset (fondamentale per evitare crash su descrittori lunghi)
    # report="S": Chiede di restituire il supporto relativo (percentuale)
    results = apriori(transactions, target=target_char, supp=supp, zmin=zmin, zmax=zmax, report="S")

    # Creazione DataFrame risultati
    res_df = pd.DataFrame(results, columns=["itemset", "support"])

    # Conversione sicura a numeri (gestisce errori eventuali) e arrotondamento
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce')
    res_df['support'] = res_df['support'].round(2)
    
    # Ordiniamo dal più frequente al meno frequente
    res_df = res_df.sort_values(by="support", ascending=False).reset_index(drop=True)

    # --- FASE 5: Stampa Risultati ---
    if print_info:
        print(f"\n{'='*20} RISULTATI TROVATI {'='*20}")
        print(f"Totale Itemset Trovati: {len(res_df)}")
        if not res_df.empty:
            print("\n--- TOP 10 ITEMSET PER SUPPORTO ---")
            # Impostazioni di stampa pandas per vedere bene le liste
            with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
                print(res_df.head(10).to_string(index=False))
        print(f"{'='*59}\n")

    return res_df

def make_apriori_association_rules(df, columns, supp, conf, zmin, zmax, target_char='r', use_descriptors=False, min_item_count=2, print_info=True):
    """
    Esegue l'algoritmo Apriori per generare Regole di Associazione (Antecedente -> Conseguente).

    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati.
        columns (list): Lista dei nomi delle colonne da analizzare (usato solo se use_descriptors=False).
        supp (float): Supporto minimo in percentuale (es. 5 per 5%). Filtra le combinazioni poco frequenti prima di cercare regole.
        conf (float): Confidenza minima in percentuale (es. 50 per 50%). La probabilità condizionata P(Conseguente|Antecedente).
        zmin (int): Dimensione minima della regola (Antecedente + Conseguente).
        zmax (int): Dimensione massima della regola. Importante per evitare tempi di calcolo esponenziali.
        target_char (str, optional): Parametro interno per PyFim, forzato a 'r' (rules). Non modificare solitamente.
        use_descriptors (bool, optional): Se True, analizza la colonna 'Description' invece delle colonne tabellari. Default False.
        min_item_count (int, optional): Rimuove elementi rari dal dataset prima dell'analisi per velocità e pulizia. Default 2.
        print_info (bool, optional): Se True, stampa il report dell'esecuzione e le top regole. Default True.

    Returns:
        pd.DataFrame: Un DataFrame ordinato per 'lift' con le colonne:
            - 'antecedent': La causa (es. "Se compri X...").
            - 'consequent': L'effetto (es. "...allora compri Y").
            - 'support': % di transazioni che contengono entrambi.
            - 'confidence': Affidabilità della regola (%).
            - 'lift': Quanto la regola è più probabile del caso (Lift > 1 indica correlazione positiva).
    """
    
    # --- FASE 1: Scelta della sorgente dati ---
    # (Identica alla funzione precedente: sceglie tra colonna Description o colonne standard)
    if use_descriptors:
        if 'Description' not in df.columns:
            raise ValueError("La colonna 'Description' non esiste nel DataFrame.")
            
        # Applichiamo ensure_list per parsare correttamente le stringhe "{'a', 'b'}"
        transactions = df['Description'].apply(ensure_list).tolist()
    else:
        # Formattazione colonne standard "Colonna=Valore"
        transactions = df[columns].apply(
            lambda row: [f"{col}={str(row[col])}" for col in columns], 
            axis=1
        ).tolist()

    # --- FASE 2: Filtraggio Item Rari (Pruning) ---
    all_items_initial = list(chain.from_iterable(transactions))
    if min_item_count > 1:
        item_counts = Counter(all_items_initial)
        valid_items = {item for item, count in item_counts.items() if count >= min_item_count}
        
        # List comprehension annidata per filtrare velocemente
        transactions = [[item for item in t if item in valid_items] for t in transactions]
        # Rimuove le transazioni diventate vuote
        transactions = [t for t in transactions if t] 

    # --- FASE 3: Esecuzione Apriori per Regole ---
    # target='r': Specifica che vogliamo regole (Rules)
    # report="SCl": Chiede Supporto (S), Confidenza (C) e Lift (l) nell'output
    results = apriori(transactions, target='r', supp=supp, conf=conf, zmin=zmin, zmax=zmax, report="SCl")
    
    if not results:
        if print_info: print("Nessuna regola trovata.")
        return pd.DataFrame()

    # --- FASE 4: Creazione DataFrame ---
    # PyFim restituisce le regole come (Conseguente, Antecedente, Supp, Conf, Lift)
    res_df = pd.DataFrame(results, columns=["consequent", "antecedent", "support", "confidence", "lift"])
    
    # Riordiniamo le colonne per renderle leggibili: Antecedente -> Conseguente
    res_df = res_df[["antecedent", "consequent", "support", "confidence", "lift"]]

    # Conversioni numeriche sicure e arrotondamenti
    res_df['support'] = pd.to_numeric(res_df['support'], errors='coerce').round(2)
    res_df['confidence'] = pd.to_numeric(res_df['confidence'], errors='coerce').round(2)
    res_df['lift'] = pd.to_numeric(res_df['lift'], errors='coerce').round(3)

    # Ordiniamo per LIFT decrescente (le regole più "interessanti" in alto)
    res_df = res_df.sort_values(by="lift", ascending=False).reset_index(drop=True)

    # --- FASE 5: Stampa dei RISULTATI ---
    if print_info:
        print(f"\n{'='*20} REGOLE DI ASSOCIAZIONE TROVATE {'='*20}")
        print(f"Totale Regole: {len(res_df)}")
        if not res_df.empty:
            print("\nTOP 10 REGOLE (per Lift):")
            temp_df = res_df.head(10).copy()
            for i, row in temp_df.iterrows():
                # Stampa formattata per leggere facilmente la regola
                print(f"{row['antecedent']} => {row['consequent']}")
                print(f"   [Supp: {row['support']}% | Conf: {row['confidence']}% | Lift: {row['lift']}]")
                print("-" * 30)
        print(f"{'='*60}\n")

    return res_df