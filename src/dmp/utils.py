import pandas as pd
import matplotlib.pyplot as plt

def check_for_column_content(df: pd.DataFrame, column_name: str, show_hist: bool = False, only_special_char: bool = False) -> tuple:
    """Funzione che controlla il contenuto di una colonna del DataFrame
    
    Input: DataFrame, nome della colonna, flag per mostrare l'istogramma, flag per restituire solo la lista dei caratteri speciali
    
    Output: lista dei caratteri presenti nella colonna e opzionalmente l'istogramma della distribuzione dei caratteri
    
    La funzione ha ancora un problema in fase di plotting per via di alcuni caratteri speciali che matplotlib non sà rappresentare,
    in ogni caso questo crea solo dei Warnings e non impedisce l'esecuzione della funzione.
    """

    char_list = []

    # Controlla se la colonna esiste nel DataFrame, altrimenti solleva un errore
    if column_name in df.columns:
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
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Trasforma la lista totale dei carratteri in un set per evitare ripetizioni
    char_list = list(set(char_list))

    return (char_list, plt if show_hist else None)
