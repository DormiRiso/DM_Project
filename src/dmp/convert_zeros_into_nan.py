import numpy as np
import pandas as pd

def convert_zeros_into_nan(df, columns):
    """
    Converte i valori uguali a 0 in NaN nelle colonne specificate di un DataFrame.
    
    Parametri:
        df (pd.DataFrame): Il DataFrame da modificare.
        columns (str o list): Il nome della colonna o lista di colonne da processare.
    
    Ritorna:
        pd.DataFrame: Il DataFrame con i valori 0 sostituiti da NaN nelle colonne specificate.
    """
    # Se è stata passata una singola colonna come stringa, la trasformiamo in lista
    if isinstance(columns, str):
        columns = [columns]

    df[columns] = df[columns].replace(0, np.nan)
    return df
