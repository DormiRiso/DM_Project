from .split_df import split_df
from .KNN import knn
from .Naive_Bayes import naive_bayes_classifier
from .decision_tree import decision_tree_classifier
from .regression import lin_regression, nonlin_regression, multiple_regression, multivariate_regression
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
    knn(df_train, df_test, num_feats = ["NumDesires", "YearPublished"], target_col="Rating", k=200, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    
    #Faccio il KNN per le colonne "Weight" e "AgeRec"(funziona bene sia con Rating che con "-d roll action")
    knn(df_train, df_test, num_feats =["Weight", "AgeRec"], target_col="Rating", k=200, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    

    # Algoritmo di Naive-Bayes per alcune colonne
    numeric_cols = ["NumWeightVotes"]
    categoric_cols = ["Family", "IsReimplementation", "Kickstarted"]
    
    naive_bayes_classifier(
        df_train, df_test, 
        num_feats=numeric_cols, 
        cat_feats=categoric_cols, 
        target_col="Rating",
        print_metrics=True, 
        make_plot=True, 
        descriptors=descriptors, 
        check_baseline=True
    )

    # 5. Decision Tree
    print("\n3. DECISION TREE")
    # Individual Decision Tree experiments
    decision_tree_classifier(
        df_train, df_test,
        num_feats=["NumDesires", "YearPublished"],
        target_col="Rating",
        max_depth=5,
        print_metrics=True,
        make_plot=True,
        descriptors=descriptors,
        check_baseline=True
    )
    
    decision_tree_classifier(
        df_train, df_test,
        num_feats=["Weight", "AgeRec"],
        target_col="Rating",
        max_depth=5,
        print_metrics=True,
        make_plot=True,
        descriptors=descriptors,
        check_baseline=True
    )
    
    # 6. Regression:
    print("\n4. REGRESSIONS")
    
    # Magari usare due colonne belle correlate prima di averle buttate via
    lin_regression(df_train, df_test, independent_col="YearPublished", dependent_col="WeightedRating")
    
    # Questi sembrano interessanti da studiare:
    nonlin_regression(df_train, df_test, independent_col="NumDesires", dependent_col="WeightedRating")
    
    # Colonna dipendente deve essere quella di prima ma possiamo giocare su quelle indip.
    multiple_regression(df_train, df_test, independent_cols=["NumDesires", "AgeRec"], dependent_col="WeightedRating", method="Linear")
    multiple_regression(df_train, df_test, independent_cols=["NumDesires", "AgeRec"], dependent_col="WeightedRating", method="KNN")
    multiple_regression(df_train, df_test, independent_cols=["NumDesires", "AgeRec"], dependent_col="WeightedRating", method="DecisionTree")

    # Non è richiesto dalla scheda quindi valutiamo
    multivariate_regression(df_train, df_test, independent_cols=["NumDesires", "Weight"], dependent_cols=["WeightedRating", "YearPublished"])

