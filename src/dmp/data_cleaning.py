import dmp.clean_description as clean_description
import dmp.clean_ordered_columns as clean_ordered_columns
import dmp.clean_suggested_players as clean_suggested_players
import dmp.convert_zeros_into_nan as convert_zeros_into_nan

def clean_df(df):
    """
        Pulisce il DataFrame e lo prepara per l'analisi.
        Input: df (DataFrame originale)
        Output: df (DataFrame pulito)
    """

    # Crea una copia del DataFrame per evitare modifiche all'originale
    df = df.copy()

    print("\n*********************************************************************\n")
    print("Inizio la pulizia del DataFrame")
    
    print("Pulisco tutte le colonne dai missing values convertendo gli 0 in NaN")
    #Converte tutti i valori 0 nel dataset come nan (solo nelle colonne in cui questo valore non ha senso)
    columns_to_convert = [
        "YearPublished", "GameWeight", "ComWeight", "MinPlayers", "MaxPlayers",
        "ComAgeRec", "LanguageEase", "BestPlayers", "MfgPlaytime",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec"
    ]
    df = convert_zeros_into_nan.convert_zeros_into_nan(df, columns_to_convert)
    
    print("Pulisco Description")
    # Trasforma la colonna "Description" in insiemi di parole (set di stringhe)
    df['Description'] = clean_description.convert_string_column_to_sets(df, 'Description')

    print("Pulisco max/min players")
    # Assicura che le colonne min e max Players siano una più piccola dell'altra. In caso di 0 sostituisce con np.NaN
    df = clean_ordered_columns.clean_ordered_columns(df, 'MinPlayers', 'MaxPlayers')

    print("Pulisco good players")
    # Effettua la pulizia della colonna good players, rimuovendo ciò che non è un numero e basandosi sull'intervallo del nummero di giocatori consentiti
    df = clean_suggested_players.clean_good_players(df, 'GoodPlayers', 'MinPlayers', 'MaxPlayers')

    print("Pulisco best players")
    # Effettua la pulizia della colonna best players, rimuovendo ciò che non è un numero o non è compreso in good players
    df = clean_suggested_players.clean_best_players(df, 'BestPlayers', 'GoodPlayers')

    print("Pulisco com min/max play time")
    # Faccio la stessa cosa per ComMinPlaytime e ComMaxPlaytime
    df = clean_ordered_columns.clean_ordered_columns(df, 'ComMinPlaytime', 'ComMaxPlaytime')

    print("Concludo la pulizia del DataFrame")
    print("\n*********************************************************************\n")
    
    return df
