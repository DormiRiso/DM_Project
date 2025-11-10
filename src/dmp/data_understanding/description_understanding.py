from dmp.utils import save_figure
from dmp.config import VERBOSE
from dmp.my_graphs import bar_graph

def count_word_occurrences(df: pd.DataFrame, column_name: str) -> dict:
    """Funzion che conta quante volte ogni parola appare in una colonna del DataFrame.
    
    Input: DataFrame, nome della colonna

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



    #plt = bar_graph()

    return word_occurrences
