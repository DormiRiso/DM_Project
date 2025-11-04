import pandas as pd

def convert_string_column_to_ints(df, col_name, map_string_to_int):
    """
    Converte una colonna di un DataFrame pandas contenente stringhe in interi,
    dopo aver verificato che tutti i valori siano tra quelli ammessi.

    Parametri:
    ----------
    df : pandas.DataFrame
        Il DataFrame su cui lavorare.

    col_name : str
        Il nome della colonna da convertire.

    map_string_to_int : dict
        Dizionario che mappa le stringhe ai valori interi desiderati.
        Es: {"low": -1, "medium": 0, "high": 1}

    Ritorna:
    --------
    pandas.DataFrame
        Il DataFrame con la colonna convertita.
    
    Solleva:
    --------
    ValueError
        Se la colonna contiene valori non presenti nelle chiavi del dizionario.
    """

    # Ottiene l'insieme dei valori unici nella colonna,
    # escludendo eventuali NaN per non interferire con il controllo
    unique_vals = set(df[col_name].dropna().unique())

    # Determina se ci sono valori non validi,
    # cioè presenti nella colonna ma non nel dizionario di mapping
    invalid_vals = unique_vals - set(map_string_to_int.keys())
    
    # Se esistono valori non ammessi, solleva un errore esplicativo
    if invalid_vals:
        raise ValueError(f"La colonna '{col_name}' contiene valori non validi: {invalid_vals}")
    
    # Se tutti i valori sono validi, applica la conversione
    # usando il metodo .map() per sostituire ogni stringa con l’intero corrispondente
    df[col_name] = df[col_name].map(map_string_to_int)
    
    # Restituisce il DataFrame modificato
    return df
