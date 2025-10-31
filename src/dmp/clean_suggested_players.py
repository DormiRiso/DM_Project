import numpy as np
import pandas as pd


def clean_good_players(df, good_players_column, lower_column, higher_column):
    """Funzione che esegue la pulizia della colonna good_player, controllando che non ci siano valori nulli o che questi 
    rientrino nell'intervallo dei giocatori consentiti
    
    Input: DataFrame, nome della colonna dei good_player, nome della colonna dei min_players, nome della colonna dei max_players
    
    Output: DataFrame con colonna good_players modificata
    
    FINIRE LOGICA DI NUMERI FUORI INTERVALLO
    """

    df = df.copy()

    # Verifica che le colonne esistano nel DataFrame
    for col in [good_players_column, lower_column, higher_column]:
        if col not in df.columns:
            raise ValueError(f"La colonna '{col}' non esiste nel DataFrame.")

    cleaned_values = []

    # Scorri il DataFrame iterando sulle righe con un indice
    for i, row in df.iterrows():
        
        entry = str(row[good_players_column])

        # Estrai solo i numeri
        numbers = [int(x) for x in entry if x.isdigit()]

        # Elimina duplicati
        unique_numbers = sorted(set(numbers))

        lower = row[lower_column]
        upper = row[higher_column]

        # Se non ho nessun numero o solo zeri aggiungi nan
        if not unique_numbers or set(unique_numbers) == {0}:
            cleaned_values.append(np.nan)
            continue

        # Se ho lower e upper NaN aggiungi direttamente i numeri trovati
        if pd.isna(lower) and pd.isna(upper):
            cleaned_values.append(unique_numbers)
            continue

        # Controlla se i valori rientrano nell'intervallo, allora aggiungi i numeri trovati, altrimenti aggiusta la lista di good_player
        if (min(unique_numbers) >= lower) and (max(unique_numbers) <= upper):
            cleaned_values.append(unique_numbers)
        else:
            cut_unique_numbers = [num for num in unique_numbers if (num >= lower) and (num <= upper)]
            cleaned_values.append(cut_unique_numbers if cut_unique_numbers else np.nan)

    # Aggiorna la colonna nel DataFrame
    df[good_players_column] = cleaned_values

    return df

def clean_best_players(df, best_player_column):
    pass