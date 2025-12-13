from sklearn.neighbors import KNeighborsClassifier
from .classification_utils import (
    prepare_target_column, 
    clean_and_process_data, 
    make_metrics, 
    run_baseline_analysis, 
    generate_plots
)

def knn(train_df, test_df, num_feats=None, cat_feats=None, target_col="Rating", k=3, print_metrics=False, make_plot=False, descriptors=None, check_baseline=False):
    """
    Funzione principale per la classificazione K-Nearest Neighbors (KNN).
    """
    
    # -------------------------------------------------------------------------
    # 1. PREPARAZIONE TARGET (Deve essere la prima cosa!)
    # -------------------------------------------------------------------------
    # Determiniamo PRIMA il nome del target e prepariamo i dataframe
    df_train_aug, actual_target = prepare_target_column(train_df, descriptors, target_col)
    df_test_aug, _ = prepare_target_column(test_df, descriptors, target_col)

    # -------------------------------------------------------------------------
    # 2. STAMPA INFO (Ora possiamo farlo perché actual_target esiste)
    # -------------------------------------------------------------------------
    # Gestione liste vuote per la stampa
    num_str = ", ".join(num_feats) if num_feats else "Nessuna"
    cat_str = ", ".join(cat_feats) if cat_feats else "Nessuna"
    
    print(
f"""
========================================================
✨ INIZIO CLASSIFICAZIONE: KNN (k={k})
========================================================
Target: {actual_target}
Feature Numeriche ({len(num_feats) if num_feats else 0}):   {num_str}
Feature Categoriche ({len(cat_feats) if cat_feats else 0}): {cat_str}
--------------------------------------------------------
"""
    )

    # -------------------------------------------------------------------------
    # 3. PROCESSING (Scaling + OHE)
    # -------------------------------------------------------------------------
    X_train, y_train, X_test, df_train_clean, df_test_clean, final_feature_names = clean_and_process_data(
        df_train_aug, df_test_aug, num_feats, cat_feats, actual_target
    )

    # Controllo di sicurezza
    if len(df_test_clean) == 0:
        print("⚠️ Nessun dato di test valido.")
        return df_test_clean

    # -------------------------------------------------------------------------
    # 4. ADDESTRAMENTO
    # -------------------------------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=k)
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
        run_baseline_analysis(model, X_train, y_train, f"KNN (k={k})")

    if actual_target in df_test_clean.columns:
        y_test = df_test_clean[actual_target]
        
        if print_metrics:
            make_metrics(y_test, predictions, f"KNN (k={k})", actual_target)
            
        if make_plot:
            generate_plots(
                model, X_train, y_train, X_test, y_test, 
                final_feature_names, 
                actual_target, descriptors, 
                model_tag=f"KNN_{k}"
            )

    return df_test_clean
