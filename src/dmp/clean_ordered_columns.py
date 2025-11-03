import numpy as np
import pandas as pd

def clean_ordered_columns(df, lower_column, higher_column):
    """
    Pulisce e corregge le colonne lower_column e higher_column in un DataFrame come segue:

    Regole:
    - Se lower_column == 0 → np.nan
    - Se higher_column == 0 → np.nan
    - Se lower_column > higher_column e entrambi > 0 → scambia i valori
    - Se un valore non è numerico o non è un numero intero (es. 3.5, 'a') → np.nan
    """

    # Copia per evitare modifiche indesiderate all'originale
    df = df.copy()

    # Coerce non-numeric to NaN
    df[lower_column] = pd.to_numeric(df[lower_column], errors='coerce', downcast='integer')
    df[higher_column] = pd.to_numeric(df[higher_column], errors='coerce', downcast='integer')

    # Sostituisce con NaN i valori numerici che non sono interi (es. 3.5)
    mask_lower_not_int = df[lower_column].notna() & (df[lower_column] % 1 != 0)
    mask_higher_not_int = df[higher_column].notna() & (df[higher_column] % 1 != 0)

    if mask_lower_not_int.any():
        df.loc[mask_lower_not_int, lower_column] = np.nan

    if mask_higher_not_int.any():
        df.loc[mask_higher_not_int, higher_column] = np.nan

    # Maschere logiche per le regole originali
    mask_swap = (
        (df[lower_column] > 0) &
        (df[higher_column] > 0) &
        (df[lower_column] > df[higher_column])
    )
    mask_min_zero = (df[lower_column] == 0)
    mask_max_zero = (df[higher_column] == 0)

    # Scambio dei valori se le condizioni sono soddisfatte
    if mask_swap.any():
        df.loc[mask_swap, [lower_column, higher_column]] = df.loc[mask_swap, [higher_column, lower_column]].values

    # Sostituzioni 0 → NaN se le condizioni sono soddisfatte
    if mask_min_zero.any():
        df.loc[mask_min_zero, lower_column] = np.nan

    if mask_max_zero.any():
        df.loc[mask_max_zero, higher_column] = np.nan

    return df
