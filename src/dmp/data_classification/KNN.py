from sklearn.neighbors import KNeighborsClassifier
from .classification_utils import (
    prepare_target_column, 
    clean_and_process_data, # Funzione per preprocessing e pulizia robusta
    make_metrics, 
    run_baseline_analysis, 
    generate_plots
)

def knn(train_df, test_df, num_feats=None, cat_feats=None, target_col="Rating", k=3, print_metrics=False, make_plot=False, descriptors=None, check_baseline=False):
    """
    Funzione principale per la classificazione K-Nearest Neighbors (KNN).

    Gestisce l'intera pipeline: preparazione, preprocessing (Scaling/OHE), addestramento, 
    predizione e valutazione statistica del modello KNN.

    Args:
        train_df (pd.DataFrame): DataFrame di addestramento.
        test_df (pd.DataFrame): DataFrame di test.
        num_feats (list): Colonna/e numeriche da scalare.
        cat_feats (list): Colonna/e categoriche da codificare (OHE).
        target_col (str): Nome della colonna target di default.
        k (int): Numero di vicini da considerare per la classificazione.
        print_metrics (bool): Se True, stampa Precision, Recall, F1 sul Test Set.
        make_plot (bool): Se True, genera la Dashboard grafica (ROC/CM/Scatter).
        descriptors (list): Se fornito, il target viene generato dinamicamente (es. da keyword).
        check_baseline (bool): Se True, esegue un test di Cross-Validation contro modelli dummy.

    Returns:
        pd.DataFrame: Il DataFrame di test arricchito con la colonna 'prediction'.
    """
    
    # 1. Preparazione Target
    # Determina il nome della colonna target finale ('actual_target') in base a 'target_col' 
    # o alla logica 'descriptors' (estrazione/combinazione di keyword).
    df_train_aug, actual_target = prepare_target_column(train_df, descriptors, target_col)
    df_test_aug, _ = prepare_target_column(test_df, descriptors, target_col)

    # 2. Processing (Scaling + OHE)
    # Questa è la fase critica per il KNN:
    # a) Pulizia NaN: Rimuove le righe incomplete.
    # b) Scaling: Standardizza le feature numeriche (CRUCIALE per KNN, che si basa sulle distanze euclidee).
    # c) One-Hot Encoding (OHE): Codifica le feature categoriche in numeri (necessario per calcolare le distanze).
    X_train, y_train, X_test, df_train_clean, df_test_clean, final_feature_names = clean_and_process_data(
        df_train_aug, df_test_aug, num_feats, cat_feats, actual_target
    )

    # Controllo di sicurezza: se il Test Set è vuoto dopo la pulizia dei dati
    if len(df_test_clean) == 0:
        print("⚠️ Nessun dato di test valido.")
        return df_test_clean

    # 3. Addestramento (Fitting)
    # KNN è un algoritmo "lazy": l'addestramento consiste semplicemente nel memorizzare l'intero dataset.
    model = KNeighborsClassifier(n_neighbors=k)
    # Il metodo .fit() memorizza la posizione di tutti i punti (X_train) e le loro etichette (y_train).
    model.fit(X_train, y_train)
    
    # 4. Predizione
    # Per ogni punto in X_test, il modello trova i k vicini più prossimi nel set di training
    # e assegna la classe più frequente.
    predictions = model.predict(X_test)
    # Aggiunge il risultato della predizione al DataFrame di test pulito.
    df_test_clean['prediction'] = predictions

    # 5. Baseline (Validazione Statistica)
    # Confronta il modello KNN con un classificatore Dummy per valutare se le performance 
    # sono statisticamente significative o casuali.
    if check_baseline:
        run_baseline_analysis(model, X_train, y_train, f"KNN (k={k})")

    # 6. Metriche e Plot (Reportistica)
    # Procediamo solo se abbiamo etichette vere nel test set per la validazione.
    if actual_target in df_test_clean.columns:
        y_test = df_test_clean[actual_target]
        
        if print_metrics:
            # Stampa le metriche di performance (Accuracy, Precision, Recall, F1).
            make_metrics(y_test, predictions, f"KNN (k={k})", actual_target)
            
        if make_plot:
            # Genera la "Dashboard" grafica completa (Scatterplot, Confusion Matrix, ROC Curve).
            generate_plots(
                model, X_train, y_train, X_test, y_test, 
                final_feature_names, # Nomi delle feature dopo OHE/Scaling per gli assi
                actual_target, descriptors, 
                model_tag=f"KNN_{k}"
            )

    return df_test_clean # Restituisce il Test Set con le previsioni.