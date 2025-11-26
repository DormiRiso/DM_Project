from .split_df import split_df
from .KNN import knn
import os

def classificate_df(df, percentuale, save_dfs=False, descriptors = None):
    """
    Funzione di base per tutto il data classification:
    - Divide il dataset in test e train
    - Opzionalmente salva i risultati
    """
    
    # 1. Dividi il dataset in test e train
    df_train, df_test = split_df(df, percentuale)

    # 2. Blocco di salvataggio (se richiesto)
    if save_dfs:
        cartella_destinazione = "data"

        # Creiamo la cartella se non esiste
        os.makedirs(cartella_destinazione, exist_ok=True)

        # Definiamo i percorsi completi
        path_train = os.path.join(cartella_destinazione, "dataset_train.csv")
        path_test = os.path.join(cartella_destinazione, "dataset_test.csv")

        # index=False serve per non salvare la colonna con i numeri di riga
        df_train.to_csv(path_train, index=False)
        df_test.to_csv(path_test, index=False)
        
        print(f"File salvati correttamente in: {cartella_destinazione}")


    #Faccio il KNN per le colonne "NumDesires" e "YearPublished" (Funziona bene con Rating)
    knn(df_train, df_test, ["NumDesires", "YearPublished"], "Rating", k=200, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    
    #Faccio il KNN per le colonne "Weight" e "AgeRec"(funziona bene sia con Rating che con "-d roll action")
    knn(df_train, df_test, ["Weight", "AgeRec"], "Rating", k=200, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    
