import dmp.clean_description as clean_description
import dmp.clean_ordered_columns as clean_ordered_columns

def clean_df(df):
    """
        Pulisce il DataFrame e lo prepara per l'analisi.
        Input: df (DataFrame originale)
        Output: df (DataFrame pulito)
    """

    # Crea una copia del DataFrame per evitare modifiche all'originale
    df = df.copy()

    # Trasforma la colonna "Description" in insiemi di parole (set di stringhe)
    df['Description'] = clean_description.convert_string_column_to_sets(df, 'Description')
    
    #Assicura che le colonne min e max Players siano una più piccola dell'altra. In caso di 0 sostituisce con np.NaN
    df = clean_ordered_columns.clean_ordered_columns(df, 'MinPlayers', 'MaxPlayers')
    #Faccio la stessa cosa per ComMinPlaytime e ComMaxPlaytime
    df = clean_ordered_columns.clean_ordered_columns(df, 'ComMinPlaytime', 'ComMaxPlaytime')
    
    return df
