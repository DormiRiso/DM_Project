from sklearn.naive_bayes import GaussianNB
# Importiamo le funzioni di utilità che gestiscono la maggior parte del preprocessing e della valutazione
from .classification_utils import (
    prepare_target_column, 
    clean_and_process_data,
    make_metrics, 
    run_baseline_analysis, 
    generate_plots
)

def naive_bayes_classifier(train_df, test_df, num_feats=None, cat_feats=None, target_col="Rating", print_metrics=False, make_plot=False, descriptors=None, check_baseline=False):
    """
    Funzione principale per la classificazione Gaussian Naive Bayes (GNB).

    Esegue l'intera pipeline di Machine Learning: preparazione dati, preprocessing
    (Scaling e One-Hot Encoding), addestramento GNB, predizione e reportistica.

    Args:
        train_df (pd.DataFrame): DataFrame di addestramento.
        test_df (pd.DataFrame): DataFrame di test.
        num_feats (list): Colonna/e numeriche da scalare.
        cat_feats (list): Colonna/e categoriche da codificare (OHE).
        target_col (str): Nome della colonna target di default.
        print_metrics (bool): Se True, stampa Precision, Recall, F1 su Test Set.
        make_plot (bool): Se True, genera la Dashboard grafica (ROC/CM/Scatter).
        descriptors (list): Se fornito, il target viene generato dinamicamente (es. da keyword).
        check_baseline (bool): Se True, esegue un test di Cross-Validation contro modelli dummy.

    Returns:
        pd.DataFrame: Il DataFrame di test arricchito con la colonna 'prediction'.
    """
    
    # 1. Preparazione Target
    # Chiama una funzione utility che gestisce la logica dei 'descriptors'.
    # Se 'descriptors' è fornito, cerca le keyword o unisce le colonne per creare la variabile target.
    # 'actual_target' contiene il nome della colonna target finale (es. "Rating" o "Desc_A - Desc_B").
    df_train_aug, actual_target = prepare_target_column(train_df, descriptors, target_col)
    df_test_aug, _ = prepare_target_column(test_df, descriptors, target_col)

    # 2. Pulizia e Processing (Scaling + OneHotEncoding)
    # Questa è la fase di Preprocessing cruciale per i dati misti.
    X_train, y_train, X_test, df_train_clean, df_test_clean, final_feature_names = clean_and_process_data(
        df_train_aug, df_test_aug, num_feats, cat_feats, actual_target
    )
    # 'clean_and_process_data' esegue:
    # a) Rimuove righe con NaN nelle features o nel target (su entrambi i set).
    # b) Applica StandardScaler (numeriche) e OneHotEncoder (categoriche) usando ColumnTransformer.
    # c) Restituisce le matrici X pronte (X_train, X_test) e i nomi delle feature trasformate.

    # Controllo di sicurezza: se non resta nulla nel test set (tutto NaN/scartato)
    if len(df_test_clean) == 0:
        print("⚠️ Nessun dato di test valido dopo la pulizia. Salto predizioni.")
        return df_test_clean

    # 3. Addestramento (Modello)
    # Inizializza il modello Gaussian Naive Bayes (GNB) 
    # GNB assume che le feature (ora tutte numeriche, grazie all'OHE e allo Scaling) seguano una distribuzione gaussiana.
    model = GaussianNB()
    # Addestra il modello utilizzando le feature scalate/codificate (X_train) e le etichette (y_train).
    model.fit(X_train, y_train)
    
    # 4. Predizione
    # Utilizza il modello addestrato per generare le previsioni sul Test Set.
    predictions = model.predict(X_test)
    # Aggiunge le previsioni al DataFrame di test pulito.
    df_test_clean['prediction'] = predictions

    # 5. Baseline (Validazione Statistica)
    # Opzionale: Confronta le performance del modello (tramite Cross-Validation)
    # con un modello "Dummy" (che sceglie sempre la classe più frequente) per verificarne la significatività.
    if check_baseline:
        run_baseline_analysis(model, X_train, y_train, "Naive Bayes")

    # 6. Metriche e Dashboard (Reportistica)
    # Controlla che la colonna target esista nel Test Set (necessario per la valutazione).
    if actual_target in df_test_clean.columns:
        y_test = df_test_clean[actual_target]
        
        if print_metrics:
            # Stampa le metriche chiave (Accuracy, Precision, Recall, F-Measure) sul Test Set.
            make_metrics(y_test, predictions, "Naive Bayes", actual_target)
            
        if make_plot:
            # Genera la "Dashboard" completa (Scatterplot, Confusion Matrix e ROC Curve).
            # Usa i nomi delle feature trasformate ('final_feature_names') per etichettare correttamente i grafici.
            generate_plots(
                model, X_train, y_train, X_test, y_test, 
                final_feature_names, 
                actual_target, descriptors, 
                model_tag="NB"
            )

    return df_test_clean # Restituisce il Test Set con le previsioni allegate.