import os
import pandas as pd
import matplotlib.pyplot as plt

def save_figure(plot, title, folder="figures", extension=".png"):

    # Crea la cartella "folder" se non esiste
    os.makedirs(folder, exist_ok=True)

    # Salvataggio del file
    file_name = title
    file_name = file_name.replace(" ", "_").lower() + extension
    file_path = os.path.join(folder, file_name)
    plot.savefig(file_path, bbox_inches='tight')
    plot.close()

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

def count_word_occurrences(df: pd.DataFrame, column_name: str, show_hist: bool = False) -> dict:
    """Funzion che conta quante volte ogni parola appare in una colonna del DataFrame.
    
    Input: DataFrame, nome della colonna, flag per mostrare l'istogramma
    
    Output: dizionario con le parole come chiavi e il numero di occorrenze come valori
    """

    word_occurrences = {}

    # Controlla se la colonna esiste nel DataFrame, altrimenti solleva un errore
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Scorre tutte le entries della colonna e conta le occorrenze delle parole
    for entry_set in df[column_name]:
        for word in entry_set:
            if word in word_occurrences:
                word_occurrences[word] += 1
            else:
                word_occurrences[word] = 1

    # Se la flag per l'istogramma è attiva plotta l'istogramma con le occorrenze per ogni parola
    if show_hist:
        plt.figure(figsize=(10, 6))
        # Per motivi di leggibilità mostriamo solo le prime 50 parole più frequenti
        sorted_words = dict(sorted(word_occurrences.items(), key=lambda item: item[1], reverse=True)[:50])
        plt.bar(sorted_words.keys(), sorted_words.values(), color='green', alpha=0.7)
        plt.title(f'Word Occurrences in Column: {column_name}')
        plt.xlabel('Words')
        plt.ylabel('Occurrences')
        plt.xticks(rotation=90)
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    return word_occurrences


def filter_column(df, colonne, by_percentile=False, by_threshold=False,
                  percentiles=(0.05, 0.95), thresholds=(1, 100)):
    """
    Filtra un DataFrame mantenendo solo le righe in cui i valori
    delle colonne specificate si trovano entro i percentili o soglie dati.

    Parametri:
    ----------
    df : pd.DataFrame
        Il DataFrame da filtrare.
    colonne : list
        Lista delle colonne su cui applicare il filtro.
    by_percentile : bool
        Se True, filtra la colonna per percentili.
    by_threshold : bool
        Se True, filtra la colonna per soglie numeriche.
    percentiles : tuple (float, float)
        Percentili inferiore e superiore (es. (0.05, 0.95) → 5° e 95° percentile).
    thresholds : tuple (float, float)
        Soglie inferiori e superiori (es. (1, 100) → valori tra 1 e 100).

    Ritorna:
    --------
    pd.DataFrame
        Un nuovo DataFrame filtrato.
    """
    df_filtrato = df.copy()
    # Se colonna non è specificata, usa tutte le colonne numeriche
    if colonne is None:
        df_filtrato = df_filtrato.select_dtypes(include=["number"])
        colonne = df_filtrato.columns.tolist()
        
    if not by_percentile and not by_threshold:
        raise ValueError("Devi specificare almeno una modalità di filtro: by_percentile=True o by_threshold=True")

    for col in colonne:
        if by_percentile:
            low, high = percentiles
            low_val = df[col].quantile(low)
            high_val = df[col].quantile(high)
            df_filtrato = df_filtrato[(df_filtrato[col] >= low_val) & (df_filtrato[col] <= high_val)]

        elif by_threshold:
            low_val, high_val = thresholds
            df_filtrato = df_filtrato[(df_filtrato[col] >= low_val) & (df_filtrato[col] <= high_val)]

    return df_filtrato


