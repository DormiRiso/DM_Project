import pandas as pd
import os
# Assicurati che queste funzioni siano importate dal file dove le hai definite
from .apriori import (
    make_apriori_for_itemsets, 
    make_apriori_association_rules, 
    analyze_apriori_sensitivity
)

def pattern_mine_df(df, run_descriptors=True):
    """
    Esegue il pattern mining.
    Genera file txt puliti nella cartella figures/pattern_mining.
    """
    
    # 1. SETUP CARTELLA OUTPUT
    output_folder = "figures/pattern_mining"
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Inizio sessione di Mining. Output folder: {output_folder}")

    # Lavoriamo su una copia
    working_df = df.copy()

    # --- 2. PRE-PROCESSING E BINNING ---
    print("--- FASE PRE-PROCESSING ---")
    
    num_columns = ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating', 'MinPlayers', 'MaxPlayers']
    
    # Imputazione mediana
    existing_num_cols = [c for c in num_columns if c in working_df.columns]
    working_df[existing_num_cols] = working_df[existing_num_cols].fillna(working_df[existing_num_cols].median())

    # Binning (Discretizzazione)
    # Fondamentale per Apriori sulle colonne numeriche
    cols_to_bin = ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating']
    for col in cols_to_bin:
        if col in working_df.columns:
            working_df[col] = pd.qcut(working_df[col], 4, duplicates='drop')
            print(f"Binning completato per: {col}")

    # --- 3. ANALISI SULLE COLONNE (STRUTTURATE) ---
    print("\n" + "="*30 + "\nANALISI: COLONNE BINNATE\n" + "="*30)
    
    # Definizione percorsi file di output per le colonne
    path_itemsets_cols = os.path.join(output_folder, "1_itemsets_columns.txt")
    path_rules_cols = os.path.join(output_folder, "2_rules_columns.txt")
    
    # A. Itemsets Frequenti
    make_apriori_for_itemsets(
        working_df, 
        columns=existing_num_cols, 
        supp=10, 
        zmin=2, zmax=5,
        target_type='frequent', 
        use_descriptors=False, 
        min_item_count=200, 
        print_info=True,
        output_file=path_itemsets_cols # Passiamo il percorso stringa
    )
    
    # B. Regole di Associazione
    make_apriori_association_rules(
        working_df, 
        columns=existing_num_cols, 
        supp=10, conf=50, 
        zmin=2, zmax=5,
        use_descriptors=False, 
        min_item_count=200, 
        print_info=True,
        output_file=path_rules_cols # Passiamo il percorso stringa
    )

    # C. Analisi Sensibilità (Grafici)
    if 'Rating' in working_df.columns:
        analyze_apriori_sensitivity(
            working_df, 
            existing_num_cols, 
            target_col='Rating', # Usiamo la colonna binnata!
            zmin=2, zmax=5, 
            use_descriptors=False, 
            min_item_count=3
        )

    
    # --- 4. ANALISI SUI DESCRITTORI (TESTUALI) ---
    if run_descriptors and 'Description' in working_df.columns:
        print("\n" + "="*30 + "\nANALISI: DESCRITTORI (Description)\n" + "="*30)
        
        # Definizione percorsi file di output per i descrittori
        path_itemsets_desc = os.path.join(output_folder, "3_itemsets_descriptors.txt")
        path_rules_desc = os.path.join(output_folder, "4_rules_descriptors.txt")
        
        # A. Itemsets Frequenti
        make_apriori_for_itemsets(
            working_df, 
            columns=[], 
            supp=5,  
            zmin=2, zmax=5,
            target_type='frequent', 
            use_descriptors=True, 
            min_item_count=1000, 
            print_info=True,
            output_file=path_itemsets_desc # Passiamo il percorso stringa
        )
        
        # B. Regole di Associazione
        make_apriori_association_rules(
            working_df, 
            columns=[], 
            supp=3, 
            conf=40, 
            zmin=2, zmax=5,
            use_descriptors=True, 
            min_item_count=1000, 
            print_info=True,
            output_file=path_rules_desc # Passiamo il percorso stringa
        )

        #CI METTE UN PAIO DI MINUTI AD ESEGUIRE
        """
        # C. Analisi Sensibilità (Grafici Logaritmici)
        if 'Rating' in working_df.columns:
            analyze_apriori_sensitivity(
                working_df, 
                existing_num_cols, 
                target_col='Rating', 
                zmin=2, zmax=5, 
                use_descriptors=True, 
                min_item_count=1000 
            )
           """ 
    print("\n--- FINE SESSIONE MINING ---")
    print(f"I file completi sono stati salvati in: {output_folder}")