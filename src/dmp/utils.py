import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def save_figure(plot, title, folder="figures", extension=".png"):

    # Crea la cartella "folder" se non esiste
    os.makedirs(folder, exist_ok=True)

    # Salvataggio del file
    file_name = title
    file_name = file_name.replace(" ", "_").lower() + extension
    file_path = os.path.join(folder, file_name)
    plot.savefig(file_path, bbox_inches='tight')

    return file_path

def check_for_column_content(df: pd.DataFrame, column_name: str, show_hist: bool = False, only_special_char: bool = False) -> tuple:
    """Funzione che controlla il contenuto di una colonna del DataFrame
    
    Input: DataFrame, nome della colonna, flag per mostrare l'istogramma, flag per restituire solo la lista dei caratteri speciali
    
    Output: lista dei caratteri presenti nella colonna e opzionalmente l'istogramma della distribuzione dei caratteri
    
    La funzione ha ancora un problema in fase di plotting per via di alcuni caratteri speciali che matplotlib non sà rappresentare,
    in ogni caso questo crea solo dei Warnings e non impedisce l'esecuzione della funzione.
    """

    char_list = []

    # Controlla se la colonna esiste nel DataFrame, altrimenti solleva un errore
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Scorre tutte le entries della colonna, le trasformarmi in string e poi aggiunge i caratteri alla lista totale
    for entry in df[column_name].astype(str):
        char_list.extend(list(entry))

    # Se la flag per i caratteri speciali è attiva rimuove dalla lista totale tutti i caratteri alfanumerici
    if only_special_char:
        char_list = [char for char in char_list if not char.isalnum()]

    # Se la flag per l'istogramma è attiva plotta l'istogramma della distribuzione dei caratteri
    if show_hist:
        plt.figure(figsize=(10, 6))
        plt.hist(char_list, bins=50, color='blue', alpha=0.7)
        plt.title(f'Character Distribution in Column: {column_name}')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    # Trasforma la lista totale dei carratteri in un set per evitare ripetizioni
    char_list = list(set(char_list))

    return (char_list, plt if show_hist else None)

def outlier_bounds_iqr(df, column, k=1.5):
    """
        Calcola i limiti inferiore e superiore per individuare outlier
        in una colonna di un DataFrame utilizzando il metodo IQR (Interquartile Range).

        Parametri
        ----------
        df : pandas.DataFrame
            Il DataFrame che contiene i dati.
        column : str
            Il nome della colonna numerica su cui calcolare i limiti degli outlier.
        k : float, opzionale (default = 1.5)
            Fattore di scala che determina quanto "larghi" devono essere i limiti.
            Valori più piccoli rendono il metodo più sensibile (più outlier trovati),
            mentre valori più grandi lo rendono più tollerante.
    """

    # Calcola il primo quartile (Q1) e terzo quartile (Q3)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    # Calcola l'intervallo interquartile (IQR = Q3 - Q1),
    IQR = Q3 - Q1
    # Calcola il limite inferiore degli outlier
    lower = Q1 - k * IQR
    # Calcola il limite superiore degli outlier
    upper = Q3 + k * IQR

    return lower, upper


def filter_columns(df, colonne=None, method="threshold", params=None, delete_row=False):
    """
    Filtra un DataFrame mantenendo solo i valori entro i limiti scelti:
      - 'percentile' → usa 'params' con chiavi ('low', 'high')
      - 'threshold'  → usa 'params' con chiavi ('low', 'high')
      - 'iqr'        → usa 'params' con chiave ('k')

    Parametri
    ----------
    df : pd.DataFrame
        Il DataFrame da filtrare.
    colonne : list, opzionale
        Colonne su cui applicare il filtro (se None → tutte quelle numeriche).
    method : str
        Metodo da utilizzare: 'percentile', 'threshold' o 'iqr'.
    params : dict, opzionale
        Dizionario di parametri specifici per il metodo scelto.
    delete_row : bool
        Se True elimina le righe con outlier; se False sostituisce solo gli outlier con NaN.

    Ritorna
    -------
    pd.DataFrame
        DataFrame filtrato secondo il metodo scelto.
    """

    df_filtrato = df.copy()

    # Se non sono specificate colonne, usa tutte le numeriche
    if colonne is None:
        colonne = df_filtrato.select_dtypes(include=["number"]).columns.tolist()

    # Controllo metodo valido
    method = method.lower()
    metodi_validi = ["iqr", "percentile", "threshold"]
    if method not in metodi_validi:
        raise ValueError(f"Metodo non valido. Scegli tra: {metodi_validi}")

    # Valori di default per params
    if params is None:
        params = {}
    low = params.get("low", None)
    high = params.get("high", None)
    k = params.get("k", 1.5)

    for col in colonne:
        if method == "percentile":
            low_val = df[col].quantile(low if low is not None else 0.05)
            high_val = df[col].quantile(high if high is not None else 0.95)
        elif method == "threshold":
            low_val = low if low is not None else df[col].min()
            high_val = high if high is not None else df[col].max()
        elif method == "iqr":
            low_val, high_val = outlier_bounds_iqr(df, col, k=k)

        if delete_row:
            # Rimuove le righe con outlier
            df_filtrato = df_filtrato[
                (df_filtrato[col] >= low_val) & (df_filtrato[col] <= high_val)
            ]
        else:
            # Sostituisce solo i valori outlier con NaN
            mask = (df_filtrato[col] < low_val) | (df_filtrato[col] > high_val)
            df_filtrato.loc[mask, col] = np.nan

    # --- Check finale: rimuovi colonne vuote e avvisa ---
    colonne_valide = [col for col in df_filtrato.columns if not df_filtrato[col].dropna().empty]
    colonne_rimosse = set(df_filtrato.columns) - set(colonne_valide)

    if colonne_rimosse:
        print(f"⚠️ Le seguenti colonne non hanno valori validi dopo il filtro e saranno rimosse: {colonne_rimosse}")

    df_filtrato = df_filtrato[colonne_valide]

    return df_filtrato




