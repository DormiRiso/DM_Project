import pandas as pd

def split_df(df, percentuale, random_seed=42):
    """
    Divide un DataFrame in due parti basandosi su una percentuale e li restituisce.

    Args:
        df (pd.DataFrame): Il DataFrame da dividere.
        percentuale (float): La percentuale per la prima parte (es. 0.8 per 80%).
        random_seed (int): (Opzionale) Seme per la riproducibilità.

    Returns:
        tuple: Una tupla contenente (df_parte_1, df_parte_2)
    """
    
    # Controllo validità percentuale
    if not 0 < percentuale < 1:
        raise ValueError("La percentuale deve essere compresa tra 0 e 1 (es. 0.7)")

    # 1. Creazione della prima parte (random sample)
    df_parte_1 = df.sample(frac=percentuale, random_state=random_seed)
    
    # 2. Creazione della seconda parte (tutto ciò che non è nella prima)
    df_parte_2 = df.drop(df_parte_1.index)

    return df_parte_1, df_parte_2