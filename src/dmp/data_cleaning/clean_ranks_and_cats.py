import pandas as pd
import numpy as np


def clean_ranks_and_cats(df: pd.DataFrame, rank_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Verifica la coerenza tra colonne Rank e Cat e crea una nuova colonna 'RanksArray'.
    
    Parametri:
        df (pd.DataFrame): Il dataframe originale.
        rank_cols (list): Lista delle colonne dei rank (es. ["Rank:thematic", "Rank:strategygames", ...]).
        cat_cols (list): Lista delle colonne binarie corrispondenti (es. ["Cat:Thematic", "Cat:Strategy", ...]).
    
    Ritorna:
        pd.DataFrame: Il dataframe con una nuova colonna 'RanksArray'.
    """
    
    # --- Verifica che le liste abbiano la stessa lunghezza ---
    if len(rank_cols) != len(cat_cols):
        raise ValueError("Le liste rank_cols e cat_cols devono avere la stessa lunghezza.")

    # --- Verifica coerenza per ogni coppia ---
    for rank_col, cat_col in zip(rank_cols, cat_cols):
        if rank_col not in df.columns or cat_col not in df.columns:
            raise KeyError(f"Colonna mancante: {rank_col} o {cat_col} non trovata nel DataFrame.")

        mask = (df[cat_col] == 1) & (df[rank_col].isna())
        if mask.any():
            print(f"Incoerenza trovata in {rank_col} / {cat_col}: {mask.sum()} righe hanno Cat=1 ma Rank=NaN.")
        
        mask = (df[cat_col] == 0) & (df[rank_col]>0)
        if mask.any():
            print(f"Incoerenza trovata in {rank_col} / {cat_col}: {mask.sum()} righe hanno Cat=0 ma Rank>0.")

    # --- Sostituiamo i NaN con 0 ---
    df[rank_cols] = df[rank_cols].fillna(0)

    # --- Creiamo la colonna con l'array dei ranks ---
    df["Ranks"] = df[rank_cols].apply(lambda x: [int(v) for v in x.values], axis=1)


    return df
