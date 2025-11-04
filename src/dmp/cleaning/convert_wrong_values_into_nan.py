import numpy as np
import pandas as pd

def convert_wrong_values_into_nan(df, columns):
    """
    Converte i valori pari a 0 o maggiori del numero di righe del DataFrame in NaN
    nelle colonne specificate.

    Parametri
    ----------
    df : pandas.DataFrame
        Il DataFrame da modificare.
    columns : str o list
        Nome della colonna o lista di colonne da processare.

    Ritorna
    -------
    pandas.DataFrame
        Il DataFrame con i valori 0 o troppo grandi sostituiti da NaN
        nelle colonne specificate.
    """
    # Se viene passata una singola colonna, trasformala in lista
    if isinstance(columns, str):
        columns = [columns]

    # Numero di righe del dataset
    n_rows = len(df)

    # Applica la sostituzione col metodo mask
    # Imposta a NaN i valori che sono == 0 oppure > n_rows
    df[columns] = df[columns].mask((df[columns] == 0) | (df[columns] > n_rows), np.nan)

    return df
