import numpy as np
import pandas as pd

def clean_ordered_columns(df, lower_column, higher_column):
    """
    Pulisce e corregge le colonne lower_column e higher_column in un DataFrame come segue:

    Regole:
    - Se lower_column == 0 → np.nan
    - Se lower_column == 0 → np.nan
    - Se lower_column > higher_column e entrambi > 0 → scambia i valori
    """

    # Copia per evitare modifiche indesiderate all'originale
    df = df.copy()

    # Maschere logiche
    mask_swap = (
        (df[lower_column] > 0) &
        (df[higher_column] > 0) &
        (df[lower_column] > df[higher_column])
    )
    mask_min_zero = (df[lower_column] == 0)
    mask_max_zero = (df[higher_column] == 0)

    # Scambio dei valori se le condizioni sono soddisfatte
    if mask_swap.any():
        df.loc[mask_swap, [lower_column, higher_column]] = (
            df.loc[mask_swap, [higher_column, lower_column]].values
        )

    # Sostituzioni 0 → NaN se le condizioni sono soddisfatte
    if mask_min_zero.any():
        df.loc[mask_min_zero, lower_column] = np.nan

    if mask_max_zero.any():
        df.loc[mask_max_zero, higher_column] = np.nan

    return df
