import pandas as pd
import numpy as np

def merge_columns_with_prefix(df, prefix, new_col):
    """
    Unisce tutte le colonne che iniziano con un certo prefisso (es. 'cat:')
    in una singola colonna che contiene array di 0/1.

    Parametri
    ----------
    df : pandas.DataFrame
        Il DataFrame da elaborare.
    prefix : str
        Prefisso usato per identificare le colonne di categoria (default: 'cat:').
    new_col : str
        Nome della nuova colonna che conterrà gli array 0/1.

    Ritorna
    -------
    pandas.DataFrame
        Il DataFrame con la nuova colonna di array e le colonne originali di categoria rimosse.
    """
    # Trova tutte le colonne che iniziano con il prefisso indicato
    cat_cols = [col for col in df.columns if col.startswith(prefix)]

    if not cat_cols:
        # Nessuna colonna trovata → restituisci il DataFrame invariato
        return df

    # Crea una nuova colonna che contiene array NumPy di 0/1
    df[new_col] = df[cat_cols].apply(lambda row: np.array(row.values, dtype=int), axis=1)

    # Rimuove le colonne originali di categoria
    df = df.drop(columns=cat_cols)

    return df
