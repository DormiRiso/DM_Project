import pandas as pd

def add_weighted_rating(df: pd.DataFrame, rating_col='Rating', votes_col='NumUserRatings', new_col='WeightedRating', m=None) -> pd.DataFrame:
    """
    Aggiunge una nuova colonna con il rating ponderato simile a IMDB/Steam.
    
    Parametri:
        df (pd.DataFrame): DataFrame contenente i voti.
        rating_col (str): Colonna con i voti medi (-1 a 1 nel tuo caso).
        votes_col (str): Colonna con il numero di voti positivi.
        new_col (str): Nome della nuova colonna da creare.
        m (int, opzionale): Numero minimo di voti considerato affidabile.
                             Se None, usa la media del numero di voti.
                             
    Ritorna:
        pd.DataFrame: DataFrame con la nuova colonna 'weighted_rating'.
    """
    R = df[rating_col]
    v = df[votes_col]
    
    C = df[rating_col].mean()  # voto medio globale
    
    if m is None:
        m = df[votes_col].mean()  # soglia media
    
    df[new_col] = (v / (v + m)) * R + (m / (v + m)) * C
    
    return df
