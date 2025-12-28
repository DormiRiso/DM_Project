import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Importiamo le tue utils esistenti
from .classification_utils import (
    prepare_target_column, 
    clean_and_process_data, 
    make_metrics, 
    run_baseline_analysis,
    _plot_separated_roc,
    _plot_knn_k_search,
    _plot_confusion_matrix,
    _plot_separated_precision_recall
)


# =============================================================================
# KNN MODIFICATO
# =============================================================================

def knn(train_df, test_df, num_feats=None, cat_feats=None, target_col="Rating", k=3, print_metrics=False, make_plot=False, descriptors=None, check_baseline=False):
    """
    Funzione principale per la classificazione K-Nearest Neighbors (KNN).
    Include plot separati e analisi del parametro K.
    """
    
    # 1. PREPARAZIONE TARGET
    df_train_aug, actual_target = prepare_target_column(train_df, descriptors, target_col)
    df_test_aug, _ = prepare_target_column(test_df, descriptors, target_col)

    # 2. STAMPA INFO
    num_str = ", ".join(num_feats) if num_feats else "Nessuna"
    cat_str = ", ".join(cat_feats) if cat_feats else "Nessuna"
    
    print(
f"""
========================================================
✨ INIZIO CLASSIFICAZIONE: KNN (k={k})
========================================================
Target: {actual_target}
Feature Numeriche: {num_str}
Feature Categoriche: {cat_str}
--------------------------------------------------------
"""
    )

    # 3. PROCESSING
    X_train, y_train, X_test, df_train_clean, df_test_clean, final_feature_names = clean_and_process_data(
        df_train_aug, df_test_aug, num_feats, cat_feats, actual_target
    )

    if len(df_test_clean) == 0:
        print("⚠️ Nessun dato di test valido.")
        return df_test_clean

    # 4. ADDESTRAMENTO
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    # 5. PREDIZIONE
    predictions = model.predict(X_test)
    df_test_clean['prediction'] = predictions

    # 6. BASELINE & METRICHE & PLOT CUSTOM
    if check_baseline:
        run_baseline_analysis(model, X_train, y_train, f"KNN (k={k})")

    if actual_target in df_test_clean.columns:
        y_test = df_test_clean[actual_target]
        
        if print_metrics:
            make_metrics(y_test, predictions, f"KNN (k={k})", actual_target)
            
        if make_plot:
            print(f"📊 Generazione e salvataggio grafici per KNN...")
            
            # NOTA: Passiamo 'final_feature_names' e 'descriptors' alle funzioni
            model_tag = f"KNN"

            # A. Confusion Matrix
            _plot_confusion_matrix(
                model, predictions, y_test, model_tag, 
                final_feature_names, descriptors, actual_target
            )
            
            # B. 3 ROC Curves separate
            _plot_separated_roc(
                model, X_test, y_test, model_tag, 
                final_feature_names, descriptors, actual_target
            )
            
            #C. 3 Accuracy-Recall plots
            _plot_separated_precision_recall(
                model, X_test, y_test, model_tag, 
                final_feature_names, descriptors, actual_target
            )
            
            # D.Plot Accuratezza vs K
            _plot_knn_k_search(
                X_train, y_train, "KNN", 
                final_feature_names, descriptors, actual_target, 
                max_k=100,
                step_k= 5 
            )

    return df_test_clean
