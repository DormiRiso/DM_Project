import dmp.clean_description as clean_description
import dmp.clean_ordered_columns as clean_ordered_columns
import dmp.clean_suggested_players as clean_suggested_players
import dmp.convert_wrong_values_into_nan as convert_wrong_values_into_nan
import dmp.remove_columns as remove_columns
import dmp.clean_ranks_and_cats as clean_ranks_and_cats
import dmp.convert_string_column_to_ints as convert_string_column_to_ints

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
    
    print("Pulisco tutte le colonne dai missing values convertendo gli 0 e i valori maggiori del numero di righe in NaN")
    #Converte tutti i valori 0 o maggiori del numero di righe nel dataset come nan (solo nelle colonne in cui questo valore non ha senso)
    columns_to_convert = [
        "YearPublished", "GameWeight", "ComWeight", "MinPlayers", "MaxPlayers",
        "ComAgeRec", "LanguageEase", "BestPlayers", "MfgPlaytime",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "Rank:strategygames",
        "Rank:abstracts", "Rank:familygames", "Rank:thematic", "Rank:cgs",
        "Rank:wargames", "Rank:partygames","Rank:childrensgames"
    ]
    df = convert_wrong_values_into_nan.convert_wrong_values_into_nan(df, columns_to_convert)

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

    #Se la colonna numComments contiene solo 0 la elimino
    # Se tutti i valori sono 0 (o la colonna è vuota)
    if (df['NumComments'] == 0).all():
        df = remove_columns.remove_columns(df, 'NumComments')
        print("Colonna NumComments eliminata, erano tutti 0")
    else:
        print("In NumComments ci sono dei valori diversi da 0")

    #Elimino la colonna ImagePath e BGGId in quanto inutili
    df = remove_columns.remove_columns(df, 'ImagePath')
    df = remove_columns.remove_columns(df, 'BGGId')
    print("Colonne ImagePath, BGGID eliminate")

    #Se c'è corrispondenza tra le colonne 'cat:' e 'Rank:' unisco le colonne 'rank:' in un vettore unico e elimino le colonne 'Cat:'
    rank_cols = [
    "Rank:strategygames", "Rank:abstracts", "Rank:familygames",
    "Rank:thematic", "Rank:cgs", "Rank:wargames",
    "Rank:partygames", "Rank:childrensgames"
    ]

    cat_cols = [
    "Cat:Strategy", "Cat:Abstract", "Cat:Family",
    "Cat:Thematic", "Cat:CGS", "Cat:War",
    "Cat:Party", "Cat:Childrens"
    ]

    df = clean_ranks_and_cats.clean_ranks_and_cats(df, rank_cols, cat_cols)
    print("Unisco le colonne 'cat' e 'rank' in una colonna di arrays chiamata 'Ranks'")
    #Rimuovo le colonne 'cat' e 'rank' dopo averle convertite in array
    df = remove_columns.remove_columns(df, rank_cols)
    df = remove_columns.remove_columns(df, cat_cols)
    print("Rimosse le colonne 'cat' e 'rank'")
    
    #Converto le colonne necessarie in interi64 (così non ho problemi con i NaN)
    Cols_to_convert_in_int = ["YearPublished", "MinPlayers", "MaxPlayers", 
    "BestPlayers", "MfgPlaytime", "ComMinPlaytime", "ComMaxPlaytime",
    "MfgAgeRec"]

    df[Cols_to_convert_in_int] = df[Cols_to_convert_in_int].astype("Int64")
    print("Converto le colonne che lo necessitano in Int64 (per i NaN)")

    #Converto la colonna Rating in interi mappandoli con -1, 0, 1
    map_string_to_int = {"Low": -1, "Medium": 0, "High": 1}
    df = convert_string_column_to_ints.convert_string_column_to_ints(df, 'Rating', map_string_to_int)
    print("Convertito la colonna 'Rating' in interi (-1,0,1)")

    #Stampo i tipi all'interno del df per fare un check veloce
    print("\n Tipi all'interno del df:")
    print(df.dtypes)


    print("Concludo la pulizia del DataFrame")
    print("\n*********************************************************************\n")
    return df
