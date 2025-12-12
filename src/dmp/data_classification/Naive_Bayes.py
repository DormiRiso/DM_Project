from sklearn.naive_bayes import GaussianNB
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
    """
    
    # -------------------------------------------------------------------------
    # 1. PREPARAZIONE TARGET
    # -------------------------------------------------------------------------
    # Determiniamo PRIMA il nome del target.
    df_train_aug, actual_target = prepare_target_column(train_df, descriptors, target_col)
    df_test_aug, _ = prepare_target_column(test_df, descriptors, target_col)
    
    # -------------------------------------------------------------------------
    # 2. STAMPA INFO (Ora possiamo farlo perché actual_target esiste)
    # -------------------------------------------------------------------------
    # Prepara le stringhe per la stampa (gestendo i casi None/lista vuota)
    num_str = ", ".join(num_feats) if num_feats else "Nessuna"
    cat_str = ", ".join(cat_feats) if cat_feats else "Nessuna"
    
    print(
f"""
========================================================
✨ INIZIO CLASSIFICAZIONE: Naive-Bayes
========================================================
Target: {actual_target}
Feature Numeriche ({len(num_feats) if num_feats else 0}):   {num_str}
Feature Categoriche ({len(cat_feats) if cat_feats else 0}): {cat_str}
--------------------------------------------------------
"""
    )

    # -------------------------------------------------------------------------
    # 3. PROCESSING (Scaling + OneHotEncoding)
    # -------------------------------------------------------------------------
    # Questa è la fase di Preprocessing cruciale per i dati misti.
    X_train, y_train, X_test, df_train_clean, df_test_clean, final_feature_names = clean_and_process_data(
        df_train_aug, df_test_aug, num_feats, cat_feats, actual_target
    )

    # Controllo di sicurezza: se non resta nulla nel test set (tutto NaN/scartato)
    if len(df_test_clean) == 0:
        print("⚠️ Nessun dato di test valido dopo la pulizia. Salto predizioni.")
        return df_test_clean

    # -------------------------------------------------------------------------
    # 4. ADDESTRAMENTO
    # -------------------------------------------------------------------------
    # GNB assume che le feature (ora tutte numeriche) seguano una distribuzione gaussiana.
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 5. PREDIZIONE
    # -------------------------------------------------------------------------
    predictions = model.predict(X_test)
    df_test_clean['prediction'] = predictions

    # -------------------------------------------------------------------------
    # 6. BASELINE & METRICHE
    # -------------------------------------------------------------------------
    if check_baseline:
        run_baseline_analysis(model, X_train, y_train, "Naive Bayes")

    if actual_target in df_test_clean.columns:
        y_test = df_test_clean[actual_target]
        
        if print_metrics:
            make_metrics(y_test, predictions, "Naive Bayes", actual_target)
            
        if make_plot:
            generate_plots(
                model, X_train, y_train, X_test, y_test, 
                final_feature_names, 
                actual_target, descriptors, 
                model_tag="NB"
            )

    return df_test_clean
