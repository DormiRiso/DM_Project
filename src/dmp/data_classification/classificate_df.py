from .split_df import split_df
from .KNN import knn
from .Naive_Bayes import naive_bayes_classifier
from .decision_tree import decision_tree_classifier
from .regression import single_regression, multiple_regression, multivariate_regression, hyperparameter_tuning_multiple, hyperparameter_tuning_single
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
    knn(df_train, df_test, num_feats = ["NumDesires", "YearPublished"], target_col="Rating", k=20, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    
    #Faccio il KNN per le colonne "Weight" e "AgeRec"(funziona bene sia con Rating che con "-d roll action")
    knn(df_train, df_test, num_feats =["Weight", "AgeRec"], target_col="Rating", k=30, print_metrics=True, make_plot=True, descriptors=descriptors, check_baseline=True)
    

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
    
    # Magari usare due colonne belle correlate prima di averle buttate via (ALCUNI ESEMPI)
    hyperparameter_tuning_single(df_train, df_test, independent_col="ComWeight", dependent_col="GameWeight")
    best_params = {'alpha_ridge': 450.0, 'alpha_lasso' : 4e-2, 'n_neighbors' : 12, 'max_depth' : 4}
    single_regression(df_train, df_test, independent_col="ComWeight", dependent_col="GameWeight", params=best_params)
    

    hyperparameter_tuning_single(df_train, df_test, independent_col="ComAgeRec", dependent_col="MfgAgeRec")
    best_params = {'alpha_ridge': 50.0, 'alpha_lasso' : 2e-3, 'n_neighbors' : 24, 'max_depth' : 5}
    single_regression(df_train, df_test, independent_col="ComAgeRec", dependent_col="MfgAgeRec", params=best_params)   
    

    hyperparameter_tuning_single(df_train, df_test, independent_col="NumWish", dependent_col="NumWant")
    best_params = {'alpha_ridge': 1500.0, 'alpha_lasso' : 0.5, 'n_neighbors' : 12, 'max_depth' : 4}
    single_regression(df_train, df_test, independent_col="NumWish", dependent_col="NumWant", params=best_params)

    # Vediamo se WeightedRating dipende solo da NumDesires:
    hyperparameter_tuning_single(df_train, df_test, independent_col="NumDesires", dependent_col="WeightedRating")
    best_params = {'alpha_ridge': 1000.0, 'alpha_lasso' : 1000.0, 'n_neighbors' : 16, 'max_depth' : 4}
    single_regression(df_train, df_test, independent_col="NumDesires", dependent_col="WeightedRating", params=best_params)

    # Aggiungiamo anche Weight per provare a migliorare sta regressione
    indip_cols = ["Weight", "NumDesires"]
    hyperparameter_tuning_multiple(df_train, df_test, independent_cols=indip_cols, dependent_col="WeightedRating")
    best_params = {'alpha_ridge': 45.0, 'alpha_lasso' : 2e-3, 'n_neighbors' : 20, 'max_depth' : 7}
    multiple_regression(df_train, df_test, independent_cols=indip_cols, dependent_col="WeightedRating", params=best_params)

    # Non è richiesto dalla scheda quindi valutiamo
    # multivariate_regression(df_train, df_test, independent_cols=["NumDesires", "Weight"], dependent_cols=["WeightedRating", "YearPublished"])

