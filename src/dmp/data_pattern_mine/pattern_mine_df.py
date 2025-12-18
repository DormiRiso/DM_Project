import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .apriori import make_apriori_for_itemsets, make_apriori_association_rules

def pattern_mine_df(df, run_columns=True, run_descriptors=True):
    """
    Esegue il pattern mining sia sulle colonne strutturate che sui descrittori.
    """
    # Lavoriamo su una copia per sicurezza
    working_df = df.copy()

    # --- 1. PRE-PROCESSING E BINNING ---
    # Definiamo le colonne su cui fare calcoli numerici
    num_columns = ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating', 'MinPlayers', 'MaxPlayers']
    
    # Imputazione mediana solo sulle colonne presenti
    existing_num_cols = [c for c in num_columns if c in working_df.columns]
    working_df[existing_num_cols] = working_df[existing_num_cols].fillna(working_df[existing_num_cols].mean())

    # Binning (Discretizzazione)
    # Usiamo duplicates='drop' per evitare errori se i dati sono troppo concentrati
    for col in ['YearPublished', 'AgeRec', 'Playtime', 'WeightedRating']:
        if col in working_df.columns:
            working_df[col] = pd.qcut(working_df[col], 4, duplicates='drop')

    # --- 2. ANALISI SULLE COLONNE (STRUTTURATE) ---
    if run_columns:
        print("\n" + "="*30 + "\nANALISI: COLONNE BINNATE\n" + "="*30)
        # Passiamo la lista delle colonne desiderate
        make_apriori_for_itemsets(
            working_df, 
            columns=existing_num_cols, 
            supp=10, 
            zmin=2, 
            zmax=5,
            target_type='frequent', 
            use_descriptors=False, # Importante: False qui
            min_item_count=200, 
            print_info=True
        )
        
        make_apriori_association_rules(
            working_df, 
            columns=existing_num_cols, 
            supp=10, 
            conf=50, 
            zmin=2, 
            zmax=5,
            use_descriptors=False, 
            min_item_count=200, 
            print_info=True
        )

    
    # --- 3. ANALISI SUI DESCRITTORI (TESTUALI) ---
    if run_descriptors and 'Description' in working_df.columns:
        print("\n" + "="*30 + "\nANALISI: DESCRITTORI (Description)\n" + "="*30)
        # Qui 'columns' può essere None o [] perché use_descriptors=True 
        make_apriori_for_itemsets(
            working_df, 
            columns=[], 
            supp=20, # Spesso i descrittori richiedono un supporto più basso
            zmin=2, 
            zmax=5,
            target_type='frequent', 
            use_descriptors=True, # Importante: True qui
            min_item_count=20, 
            print_info=True
        )
        
        make_apriori_association_rules(
            working_df, 
            columns=[], 
            supp=20, 
            conf=40, 
            zmin=2, 
            zmax=5,
            use_descriptors=True, 
            min_item_count=20, 
            print_info=True
        )
    

