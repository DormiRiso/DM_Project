import pandas as pd
import os
import numpy as np # Serve per gestire eventuali errori numpy

from .pm_algorithms import do_pattern_mining_for_itemsets, find_association_rules, analyze_sensitivity

def pattern_mine_df(df, df_ref, output_folder="figures/pattern_mining", 
                    num_bins=5, run_descriptors=True, 
                    col_map=None):
    """
    Esegue la pipeline completa di Pattern Mining.
    Usa df_ref e col_map per calcolare i bin sui valori reali.
    """
    
    # Se l'utente non passa nulla, crea un dizionario vuoto
    if col_map is None:
        col_map = {}
    
    # --- 1. SETUP AMBIENTE ---
    print(f"\n{'#'*60}")
    print(f" AVVIO SESSIONE DI PATTERN MINING")
    print(f" Output folder: {output_folder}")
    print(f"{'#'*60}\n")

    # Creazione cartelle
    folder_apriori = os.path.join(output_folder, "apriori")
    folder_fpgrowth = os.path.join(output_folder, "fpgrowth")
    
    os.makedirs(folder_apriori, exist_ok=True)
    os.makedirs(folder_fpgrowth, exist_ok=True)

    # Lavoriamo su una copia
    working_df = df.copy()

    # --- 2. PRE-PROCESSING E DISCRETIZZAZIONE ---
    print("--- FASE 2: PRE-PROCESSING DATI ---")
    
    num_columns = ['YearPublished', 'AgeRec', 'Playtime', 'LanguageEase']
    cat_columns = ['IsReimplementation', 'Kickstarted', 'Rating', 'MinPlayers', 'MaxPlayers']
    
    # 2.1 Gestione Valori Mancanti (Imputazione)
    existing_num_cols = [c for c in num_columns if c in working_df.columns]
    working_df[existing_num_cols] = working_df[existing_num_cols].fillna(working_df[existing_num_cols].median())
    
    existing_cat_cols = [c for c in cat_columns if c in working_df.columns]
    for col in existing_cat_cols:
        working_df[col] = working_df[col].fillna("Unknown").astype(str)
    
    print(f" > Imputazione completata: {len(existing_num_cols)} num, {len(existing_cat_cols)} cat.")

    # =========================================================================
    # 2.2 BINNING (Logica corretta con MAPPING e REF)
    # =========================================================================
    cols_to_bin = ['YearPublished', 'AgeRec', 'Playtime', 'LanguageEase']
    
    for col in cols_to_bin:
        if col in working_df.columns:
            # 1. Recuperiamo il nome della colonna nel dataset di riferimento (Mapping)
            ref_col_name = col_map.get(col, col) 

            try:
                # 2. Decidiamo quale serie usare per calcolare i tagli (reference o fallback)
                if ref_col_name in df_ref.columns:
                    print(f" > Binning '{col}' usando ref '{ref_col_name}'...")
                    target_series_for_cuts = df_ref[ref_col_name]
                else:
                    print(f" > Binning '{col}' (ref '{ref_col_name}' non trovato, uso dati locali)...")
                    target_series_for_cuts = working_df[col]

                # Riempiamo i NaN temporaneamente per il calcolo
                target_series_for_cuts = target_series_for_cuts.fillna(target_series_for_cuts.median())

                # 3. Calcoliamo i bin (qcut)
                binned_series_ref = pd.qcut(target_series_for_cuts, num_bins, duplicates='drop')

                # 4. Funzione di formattazione (trasforma intervalli in stringhe leggibili)
                def format_interval(x):
                    if pd.isnull(x): return "Unknown"
                    # Se i bordi sono interi (es. Anno), niente decimali.
                    if hasattr(x, 'left') and hasattr(x, 'right'):
                        if x.left % 1 == 0 and x.right % 1 == 0:
                            return f"{int(x.left)}-{int(x.right)}"
                        return f"{x.left:.2f}-{x.right:.2f}"
                    return str(x)

                readable_bins = binned_series_ref.apply(format_interval)

                # 5. Assegnazione sicura tramite indice
                working_df[col] = readable_bins.loc[working_df.index].astype(str)
                
                print(f"   > Completato. Esempio: {working_df[col].iloc[0]}")

            except Exception as e:
                print(f" > Errore binning '{col}': {e}")
                # Fallback: metodo semplice
                try:
                    bins = pd.qcut(working_df[col], num_bins, labels=False, duplicates='drop') + 1
                    working_df[col] = bins.astype(str)
                    print("   > Fallback applicato (bin 1-5)")
                except:
                    print("   > Fallback fallito.")

    # 2.3 Formattazione Categoriche
    for col in existing_cat_cols:
        working_df[col] = working_df[col].astype(str)
    
    all_cols = existing_num_cols + existing_cat_cols

    # =========================================================================
    # PARTE 3: ALGORITMO FP-GROWTH
    # =========================================================================

    # --- 3.1 FP-GROWTH SU COLONNE STRUTTURATE ---
    print(f"\n{'-'*50}")
    print(f" ALGORITMO: FP-GROWTH | TARGET: COLONNE STRUTTURATE")
    print(f"{'-'*50}")
    
    path_itemsets_cols_fp = os.path.join(folder_fpgrowth, "itemsets_columns.txt")
    path_rules_cols_fp = os.path.join(folder_fpgrowth, "rules_columns.txt")
    
    # A. Itemsets Frequenti
    do_pattern_mining_for_itemsets(
        working_df, 
        columns=all_cols, 
        supp=10, 
        zmin=2, zmax=5,
        target_type='frequent', 
        use_descriptors=False, 
        min_item_count=1, 
        print_info=True,
        output_file=path_itemsets_cols_fp,
        algo='fpgrowth'
    )
    
    # B. Regole di Associazione
    find_association_rules(
        working_df, 
        columns=all_cols, 
        supp=10, conf=50, 
        zmin=2, zmax=5,
        use_descriptors=False, 
        min_item_count=1, 
        print_info=True,
        output_file=path_rules_cols_fp,
        algo='fpgrowth'
    )

    # C. Analisi di Sensibilità
    if 'Rating' in working_df.columns:
        print(" > Generazione grafici sensibilità (FP-Growth - Colonne)...")
        analyze_sensitivity(
            working_df, 
            all_cols, 
            target_col='Rating', 
            zmin=2, zmax=5, 
            use_descriptors=False, 
            min_item_count=1,
            algo='fpgrowth'
        )
    
    # --- 3.2 FP-GROWTH SU DESCRITTORI (TESTO) ---
    if run_descriptors and 'Description' in working_df.columns:
        print(f"\n{'-'*50}")
        print(f" ALGORITMO: FP-GROWTH | TARGET: DESCRITTORI (TEXT)")
        print(f"{'-'*50}")
        
        path_itemsets_desc_fp = os.path.join(folder_fpgrowth, "itemsets_descriptors.txt")
        path_rules_desc_fp = os.path.join(folder_fpgrowth, "rules_descriptors.txt")
        
        # A. Itemsets Frequenti
        do_pattern_mining_for_itemsets(
            working_df, 
            columns=[], 
            supp=10,  
            zmin=2, zmax=5,
            target_type='frequent', 
            use_descriptors=True, 
            min_item_count=1, 
            print_info=True,
            output_file=path_itemsets_desc_fp,
            algo='fpgrowth'
        )
        
        # B. Regole di Associazione
        find_association_rules(
            working_df, 
            columns=all_cols, 
            supp=10, 
            conf=10, 
            zmin=2, zmax=5,
            use_descriptors=True, 
            min_item_count=1, 
            print_info=True,
            output_file=path_rules_desc_fp,
            algo='fpgrowth'
        )
        
        # C. Analisi Sensibilità
        if 'Rating' in working_df.columns:
            print(" > Generazione grafici sensibilità (FP-Growth - Descrittori)...")
            analyze_sensitivity(
                working_df, 
                all_cols, 
                target_col='Rating', 
                zmin=2, zmax=5, 
                use_descriptors=True, 
                min_item_count=1000,
                algo='fpgrowth'
            )
            
    print(f"\n{'#'*60}")
    print(f" SESSIONE COMPLETATA CON SUCCESSO")
    print(f" Tutti i file sono stati salvati in: {output_folder}")
    print(f"{'#'*60}\n")
