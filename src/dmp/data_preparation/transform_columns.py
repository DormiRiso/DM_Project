import pandas as pd
import numpy as np

def min_max_scaling(df: pd.DataFrame, columns: list, feature_range=(0, 1)) -> pd.DataFrame:
    """
    Normalizza le colonne specificate del DataFrame in scala Min-Max.
    
    Parametri:
        df (pd.DataFrame): Dataset originale.
        columns (list): Lista di colonne da normalizzare.
        feature_range (tuple): Range desiderato (min, max), default (0,1).
    
    Ritorna:
        pd.DataFrame: DataFrame con colonne normalizzate.
    """
    df_scaled = df.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        if col_max != col_min:
            df_scaled[col] = (df_scaled[col] - col_min) / (col_max - col_min)  # scala 0-1
            df_scaled[col] = df_scaled[col] * (max_val - min_val) + min_val
        else:
            df_scaled[col] = min_val  # se tutti i valori sono uguali
            
    return df_scaled

def z_score_scaling(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Standardizza le colonne specificate usando lo Z-score.
    
    Parametri:
        df (pd.DataFrame): Dataset originale.
        columns (list): Lista di colonne da standardizzare.
    
    Ritorna:
        pd.DataFrame: DataFrame con colonne standardizzate.
    """
    df_scaled = df.copy()
    
    for col in columns:
        mean = df_scaled[col].mean()
        std = df_scaled[col].std()
        if std != 0:
            df_scaled[col] = (df_scaled[col] - mean) / std
        else:
            df_scaled[col] = 0  # se deviazione standard = 0
            
    return df_scaled

def log_transform(df: pd.DataFrame, columns, base: float = np.e) -> pd.DataFrame:
    """
    Converte una o più colonne in scala logaritmica.
    
    Parametri:
        df (pd.DataFrame): Dataset originale.
        columns (str o list): Colonna singola (stringa) o lista di colonne da trasformare.
        base (float): Base del log (default euleriana, np.e)
    
    Ritorna:
        pd.DataFrame: DataFrame con le colonne trasformate in log.
    """
    df_transformed = df.copy()
    
    # Normalizza l'input in lista
    if isinstance(columns, str):
        columns = [columns]
    
    for col in columns:
        if col not in df_transformed.columns:
            raise KeyError(f"La colonna '{col}' non esiste nel DataFrame")
        # Sostituisce eventuali 0 con NaN per evitare log(0)
        df_transformed[col] = np.log(df_transformed[col].replace(0, np.nan) + 1) / np.log(base)
    
    return df_transformed
