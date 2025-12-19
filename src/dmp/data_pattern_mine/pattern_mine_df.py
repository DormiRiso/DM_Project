import pandas as pd
import os

from .pm_algorithms import do_pattern_mining_for_itemsets, find_association_rules, analyze_sensitivity

def pattern_mine_df(df, output_folder="figures/pattern_mining", run_descriptors=True):
    """
    Esegue la pipeline completa di Pattern Mining (Apriori e FP-Growth).
    Analizza sia le colonne strutturate (Rating, Anno, ecc.) che i descrittori testuali.
    Genera report testuali e grafici nella cartella di output specificata.
    """
    
    # --- 1. SETUP AMBIENTE ---
    print(f"\n{'#'*60}")
    print(f" AVVIO SESSIONE DI PATTERN MINING")
    print(f" Output folder: {output_folder}")
    print(f"{'#'*60}\n")

    # Creazione struttura cartelle (divide i risultati per algoritmo per ordine)
    folder_apriori = os.path.join(output_folder, "apriori")
    folder_fpgrowth = os.path.join(output_folder, "fpgrowth")
    
    os.makedirs(folder_apriori, exist_ok=True)
    os.makedirs(folder_fpgrowth, exist_ok=True)

    # Lavoriamo su una copia per non sporcare il DataFrame originale
    working_df = df.copy()

    # --- 2. PRE-PROCESSING E DISCRETIZZAZIONE ---
    print("--- FASE 2: PRE-PROCESSING DATI ---")
    
    num_columns = ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating', 'MinPlayers', 'MaxPlayers']
    
    # 2.1 Gestione Valori Mancanti (Imputazione)
    existing_num_cols = [c for c in num_columns if c in working_df.columns]
    working_df[existing_num_cols] = working_df[existing_num_cols].fillna(working_df[existing_num_cols].median())
    print(f" > Imputazione mediana completata su {len(existing_num_cols)} colonne.")

    # 2.2 Binning (Discretizzazione)
    # Trasforma valori continui (es. 7.5, 8.1) in categorie (es. "Alto", "Medio")
    # Questo passaggio è fondamentale per far funzionare gli algoritmi di associazione.
    cols_to_bin = ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating']
    for col in cols_to_bin:
        if col in working_df.columns:
            working_df[col] = pd.qcut(working_df[col], 4, duplicates='drop')
            print(f" > Binning completato per la colonna: '{col}'")


    # =========================================================================
    # PARTE 3: ALGORITMO APRIORI
    # =========================================================================

    # --- 3.1 APRIORI SU COLONNE STRUTTURATE ---
    print(f"\n{'-'*50}")
    print(f" ALGORITMO: APRIORI | TARGET: COLONNE STRUTTURATE")
    print(f"{'-'*50}")
    
    path_itemsets_cols = os.path.join(folder_apriori, "itemsets_columns.txt")
    path_rules_cols = os.path.join(folder_apriori, "rules_columns.txt")
    
    # A. Estrazione Itemsets Frequenti
    do_pattern_mining_for_itemsets(
        working_df, 
        columns=existing_num_cols, 
        supp=10, 
        zmin=2, zmax=5,
        target_type='frequent', 
        use_descriptors=False, 
        min_item_count=3, 
        print_info=True,
        output_file=path_itemsets_cols,
        algo='apriori'
    )
    
    # B. Estrazione Regole di Associazione
    find_association_rules(
        working_df, 
        columns=existing_num_cols, 
        supp=10, conf=50, 
        zmin=2, zmax=5,
        use_descriptors=False, 
        min_item_count=3, 
        print_info=True,
        output_file=path_rules_cols,
        algo='apriori'
    )

    # C. Analisi di Sensibilità (Grafici)
    if 'Rating' in working_df.columns:
        print(" > Generazione grafici sensibilità (Apriori - Colonne)...")
        analyze_sensitivity(
            working_df, 
            existing_num_cols, 
            target_col='Rating',
            zmin=2, zmax=5, 
            use_descriptors=False, 
            min_item_count=3,
            algo='apriori'
        )

    # --- 3.2 APRIORI SU DESCRITTORI (TESTO) ---
    if run_descriptors and 'Description' in working_df.columns:
        print(f"\n{'-'*50}")
        print(f" ALGORITMO: APRIORI | TARGET: DESCRITTORI (TEXT)")
        print(f"{'-'*50}")
        
        path_itemsets_desc = os.path.join(folder_apriori, "itemsets_descriptors.txt")
        path_rules_desc = os.path.join(folder_apriori, "rules_descriptors.txt")
        
        # A. Itemsets Frequenti
        do_pattern_mining_for_itemsets(
            working_df, 
            columns=[], 
            supp=10,  
            zmin=2, zmax=5,
            target_type='frequent', 
            use_descriptors=True, 
            min_item_count=500, 
            print_info=True,
            output_file=path_itemsets_desc,
            algo='apriori'
        )
        
        # B. Regole di Associazione
        find_association_rules(
            working_df, 
            columns=[], 
            supp=10, 
            conf=50, 
            zmin=2, zmax=5,
            use_descriptors=True, 
            min_item_count=500, 
            print_info=True,
            output_file=path_rules_desc,
            algo='apriori'
        )

        # NOTA: L'analisi sensibilità Apriori su descrittori è molto lenta (qualche minuto)
        """
        if 'Rating' in working_df.columns:
            analyze_sensitivity(
                working_df, existing_num_cols, target_col='Rating', 
                zmin=2, zmax=5, use_descriptors=True, min_item_count=500,
                algo='apriori'
            )
        """


    # =========================================================================
    # PARTE 4: ALGORITMO FP-GROWTH (Più veloce ed efficiente)
    # =========================================================================

    # --- 4.1 FP-GROWTH SU COLONNE STRUTTURATE ---
    print(f"\n{'-'*50}")
    print(f" ALGORITMO: FP-GROWTH | TARGET: COLONNE STRUTTURATE")
    print(f"{'-'*50}")
    
    path_itemsets_cols_fp = os.path.join(folder_fpgrowth, "itemsets_columns.txt")
    path_rules_cols_fp = os.path.join(folder_fpgrowth, "rules_columns.txt")
    
    # A. Itemsets Frequenti
    do_pattern_mining_for_itemsets(
        working_df, 
        columns=existing_num_cols, 
        supp=10, 
        zmin=2, zmax=5,
        target_type='frequent', 
        use_descriptors=False, 
        min_item_count=3, 
        print_info=True,
        output_file=path_itemsets_cols_fp,
        algo='fpgrowth'
    )
    
    # B. Regole di Associazione
    find_association_rules(
        working_df, 
        columns=existing_num_cols, 
        supp=10, conf=50, 
        zmin=2, zmax=5,
        use_descriptors=False, 
        min_item_count=3, 
        print_info=True,
        output_file=path_rules_cols_fp,
        algo='fpgrowth'
    )

    # C. Analisi di Sensibilità
    if 'Rating' in working_df.columns:
        print(" > Generazione grafici sensibilità (FP-Growth - Colonne)...")
        analyze_sensitivity(
            working_df, 
            existing_num_cols, 
            target_col='Rating', 
            zmin=2, zmax=5, 
            use_descriptors=False, 
            min_item_count=3,
            algo='fpgrowth'
        )

    
    # --- 4.2 FP-GROWTH SU DESCRITTORI (TESTO) ---
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
            min_item_count=500, 
            print_info=True,
            output_file=path_itemsets_desc_fp,
            algo='fpgrowth'
        )
        
        # B. Regole di Associazione
        find_association_rules(
            working_df, 
            columns=[], 
            supp=10, 
            conf=50, 
            zmin=2, zmax=5,
            use_descriptors=True, 
            min_item_count=500, 
            print_info=True,
            output_file=path_rules_desc_fp,
            algo='fpgrowth'
        )

        # C. Analisi Sensibilità (Ci mette un paio di minuti)
        """
        if 'Rating' in working_df.columns:
            print(" > Generazione grafici sensibilità (FP-Growth - Descrittori)...")
            print("   (Questa operazione potrebbe richiedere alcuni minuti)")
            
            analyze_sensitivity(
                working_df, 
                existing_num_cols, 
                target_col='Rating', 
                zmin=2, zmax=5, 
                use_descriptors=True, 
                min_item_count=500,
                algo='fpgrowth'
            )
        """

    print(f"\n{'#'*60}")
    print(f" SESSIONE COMPLETATA CON SUCCESSO")
    print(f" Tutti i file sono stati salvati in: {output_folder}")
    print(f"{'#'*60}\n")