import pandas as pd
import numpy as np # Serve per la permutazione casuale
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name

def knn(train_df, test_df, feature_cols, target_col, k=3, print_metrics=False, make_plot=False, descriptors=None, check_baseline=False):
    """
    Esegue k-NN. Gestisce automaticamente i NaN.
    Opzionalmente (check_baseline=True) confronta il modello con il caso puro.
    """
    
    # --- Assegnazione variabili e Copy ---
    df_train = train_df.copy()
    df_test = test_df.copy()

    # --- LOGICA DESCRITTORI vs TARGET ---
    if descriptors and isinstance(descriptors, list) and len(descriptors) > 0:
        if not set(descriptors).issubset(df_train.columns):
             raise ValueError(f"I descrittori {descriptors} non sono presenti nel df di training.")
        
        actual_target_col = " - ".join(descriptors) 
        
        # Creazione colonna combinata
        df_train[actual_target_col] = df_train[descriptors].astype(str).agg(' - '.join, axis=1)
        
        if set(descriptors).issubset(df_test.columns):
            df_test[actual_target_col] = df_test[descriptors].astype(str).agg(' - '.join, axis=1)
            
        print(f"--> Modalità Descrittori attivata. Target impostato su: '{actual_target_col}'")
    else:
        actual_target_col = target_col

    # --- RIMOZIONE NaN ---
    # 1. Pulizia Training Set
    cols_needed_train = feature_cols + [actual_target_col]
    
    if not set(cols_needed_train).issubset(df_train.columns):
         raise ValueError(f"Colonne mancanti nel training set: {set(cols_needed_train) - set(df_train.columns)}")

    initial_len = len(df_train)
    df_train.dropna(subset=cols_needed_train, inplace=True)
    
    if len(df_train) < initial_len:
        print(f"⚠️  ATTENZIONE: Rimosse {initial_len - len(df_train)} righe dal Training Set contenenti NaN.")
        if len(df_train) == 0: raise ValueError("Errore: Il dataset di Training è vuoto.")

    # 2. Pulizia Test Set (solo feature necessarie)
    df_test.dropna(subset=feature_cols, inplace=True)

    # --- Preparazione X e y ---
    X_train = df_train[feature_cols]
    y_train = df_train[actual_target_col]
    X_test = df_test[feature_cols]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Addestramento
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    
    # Predizione
    predictions = knn_model.predict(X_test_scaled)
    df_test['prediction'] = predictions
    
    # --- C. ANALISI BASELINE E INCERTEZZA ---
    if check_baseline:
        print(f"\n--- 📊 ANALISI COMPARATIVA E INCERTEZZA (k={k}) ---")
        # Usiamo Cross Validation a 5 fold sul Training Set
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # 1. KNN Reale (Cross-Validation)
        real_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        # 2. Dummy Classifier (Scommette sempre sul rating più presente)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy_scores = cross_val_score(dummy, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        # 3. Permutation Test (Etichette a caso)
        y_random = np.random.permutation(y_train)
        random_scores = cross_val_score(knn_model, X_train_scaled, y_random, cv=cv, scoring='accuracy')

        print(f"Target: {actual_target_col}")
        print(f"1. KNN Reale (CV):           {real_scores.mean():.4f} (± {real_scores.std()*2:.4f})")
        print(f"2. Baseline (Maggioranza):   {dummy_scores.mean():.4f} (± {dummy_scores.std()*2:.4f})")
        print(f"3. Random (Caso puro):       {random_scores.mean():.4f} (± {random_scores.std()*2:.4f})")
        
        diff = real_scores.mean() - dummy_scores.mean()
        if diff > 0.05: print("✅ Il modello batte la baseline in modo significativo.")
        elif diff > 0:  print("⚠️ Il modello è appena sopra la baseline (performance debole).")
        else:           print("❌ Il modello non batte la statistica di base.")
        print("-" * 50)

    # --- A. GESTIONE METRICHE ---
    if print_metrics:
        if actual_target_col in df_test.columns:
            y_true_raw = df_test[actual_target_col]
            
            # FIX IMPORTANTE: Filtriamo i NaN dal target per le metriche
            mask_valid = y_true_raw.notna()
            y_true_clean = y_true_raw[mask_valid]
            predictions_clean = predictions[mask_valid]
            
            if len(y_true_clean) > 0:
                acc = accuracy_score(y_true_clean, predictions_clean)
                prec = precision_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                rec = recall_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                f1 = f1_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                
                print(f"\n--- Performance del Modello KNN(k={k}) sul Test Set ---")
                print(f"Target:    {actual_target_col}")
                print(f"Accuracy:  {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall:    {rec:.4f}")
                print(f"F-Measure: {f1:.4f}")
                print("---------------------------------------")
            else:
                 print("ATTENZIONE: Nessun dato valido per calcolare le metriche (tutti i target sono NaN).")
        else:
            print(f"ATTENZIONE: Impossibile calcolare le metriche. Colonna '{actual_target_col}' assente nel test.")

    # --- B. GESTIONE PLOT ---
    if make_plot:
        desc_name = make_safe_descriptor_name(descriptors) if descriptors else str(target_col)
        base_dir = "figures/classification"
        os.makedirs(base_dir, exist_ok=True)
        output_path = f"{base_dir}/KNN_{desc_name}.png"

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_train,
            x=feature_cols[0], 
            y=feature_cols[1], 
            hue=actual_target_col, 
            palette='viridis',
            style=actual_target_col,
            s=100,
            edgecolor='black',
            alpha=0.8
        )
        plt.title(f'Distribuzione Classi Training\nTarget: {actual_target_col}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Classi", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path) 
        # plt.close() # Utile per liberare memoria

    return df_test