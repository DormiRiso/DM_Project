import dmp.clean_description as clean_description

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

    return df
