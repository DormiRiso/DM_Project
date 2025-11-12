import re
import ast
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

        # Se la riga è una stringa che rappresenta un set o lista, prova a convertirla
        if isinstance(entry, str):
            try:
                entry = ast.literal_eval(entry)
            except Exception:
                entry = entry.split()

        # Assicurati che sia iterabile
        if isinstance(entry, (set, list, tuple)):
            words = entry
        else:
            words = str(entry).split()

        # Conta le parole pulite
        for word in words:
            # Converti in minuscolo e tieni solo caratteri alfanumerici
            cleaned_word = re.sub(r"[^a-z0-9]+", "", word.lower())
            if cleaned_word:
                word_occurrences[cleaned_word] = word_occurrences.get(cleaned_word, 0) + 1

    # Ordina per frequenza
    sorted_words = sorted(word_occurrences.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:top_n])

    # Crea il grafico
    plt = bar_graph(
        x_data=list(top_words.keys()),
        y_data=list(top_words.values()),
        title=f"{top_n} parole più usate in '{column_name}'",
        x_label="Parole",
        y_label="Occorrenze",
        size=(12, 6),
        log_scale=False
    )

    file_path = save_figure(plt, "Occorrenza delle parole più usate nelle descrizioni", "figures/top_words", extension=".png")

    if VERBOSE:
        print(f"Grafico delle top words salvato in: {file_path}")

    return top_words
