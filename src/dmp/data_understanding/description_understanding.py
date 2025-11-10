import pandas as pd
from dmp.utils import save_figure
from dmp.config import VERBOSE
from dmp.my_graphs import bar_graph

def count_word_occurrences(df: pd.DataFrame, column_name: str, top_n: int = 50):
    """Conta le occorrenze delle parole in una colonna di un DataFrame e mostra un grafico con le prime top_n.

    Input:
        df: DataFrame contenente i dati
        column_name: nome della colonna da analizzare
        top_n: numero di parole più frequenti da visualizzare (default 50)

    Output:
        dizionario ordinato con le parole e il numero di occorrenze
    """

    word_occurrences = {}

    for entry in df[column_name]:
        if pd.isna(entry):
            continue

        # Se la riga è una stringa che rappresenta un set o lista, la convertiamo in oggetto Python
        if isinstance(entry, str):
            try:
                entry = ast.literal_eval(entry)
            except Exception:
                # Se non è parsabile come set/list, la dividiamo come testo
                entry = entry.split()

        # Ora ci assicuriamo che sia iterabile
        if isinstance(entry, (set, list, tuple)):
            words = entry
        else:
            words = str(entry).split()

        # Conta le parole
        for word in words:
            word = word.strip().lower()
            if word:
                word_occurrences[word] = word_occurrences.get(word, 0) + 1

    # Ordina per frequenza decrescente
    sorted_words = sorted(word_occurrences.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:top_n])

    # Crea il grafico con bar_graph
    plt = bar_graph(
        x_data=list(top_words.keys()),
        y_data=list(top_words.values()),
        title=f"Top {top_n} Words in '{column_name}'",
        x_label="Word",
        y_label="Occurrences",
        size=(12, 6),
        log_scale=False
    )

    file_path = save_figure(plt, "Occorrenza delle parole più usate nelle descrizioni", "figures/top_words", extension=".png")

    # Salva il grafico se richiesto
    if VERBOSE:
        print(f'Grafico delle top words salvato in: {file_path}')

    return top_words
