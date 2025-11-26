import pandas as pd
import numpy as np 
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
    Esegue una pipeline completa di classificazione K-Nearest Neighbors (K-NN).

    La funzione gestisce automaticamente:
    1. Creazione del target: Da colonna esistente o combinando descrittori (anche estraendoli da testo).
    2. Pulizia dati: Rimozione righe con NaN (KNN non gestisce valori mancanti).
    3. Scaling: Standardizzazione delle feature (z-score) fondamentale per le distanze euclidee.
    4. Validazione Statistica: Confronto (opzionale) con Baseline (Dummy) e Random Permutation.
    5. Metriche e Plot: Calcolo performance su Test Set e salvataggio scatterplot.

    Args:
        train_df (pd.DataFrame): DataFrame di addestramento.
        test_df (pd.DataFrame): DataFrame di test.
        feature_cols (list): Lista dei nomi delle colonne da usare come features (X).
        target_col (str): Nome della colonna target (y). Usata se 'descriptors' è None.
        k (int, optional): Numero di vicini per il K-NN. Default: 3.
        print_metrics (bool, optional): Se True, stampa Accuracy, Precision, Recall, F1 sul test set.
        make_plot (bool, optional): Se True, genera e salva uno scatterplot delle prime 2 feature.
        descriptors (list, optional): Lista di keyword/descrittori. Se fornita, il target viene 
                                      generato combinando questi descrittori.
        check_baseline (bool, optional): Se True, esegue Cross-Validation confrontando il modello
                                         con un DummyClassifier e un Random Permutation test.

    Returns:
        pd.DataFrame: Il DataFrame di test arricchito con la colonna 'prediction'.
    """
    
    # --- Assegnazione variabili e Copy ---
    # Creiamo copie per evitare di modificare i DataFrame originali (evita SettingWithCopyWarning)
    df_train = train_df.copy()
    df_test = test_df.copy()

    # ==============================================================================
    # --- LOGICA DESCRITTORI vs TARGET ---
    # ==============================================================================
    # Se 'descriptors' è popolato, il target non è una colonna semplice ma una derivazione
    if descriptors and isinstance(descriptors, list) and len(descriptors) > 0:
        actual_target_col = " - ".join(descriptors) 

        # CASO A: I descrittori sono colonne già esistenti nel DataFrame
        if set(descriptors).issubset(df_train.columns):
            # Unisce i valori delle colonne (es. "Alto" + "Veloce" -> "Alto - Veloce")
            df_train[actual_target_col] = df_train[descriptors].astype(str).agg(' - '.join, axis=1)
            
            # Applica la stessa logica al test set se le colonne esistono
            if set(descriptors).issubset(df_test.columns):
                df_test[actual_target_col] = df_test[descriptors].astype(str).agg(' - '.join, axis=1)
            print(f"--> Modalità Descrittori (Colonne). Target: '{actual_target_col}'")

        # CASO B: Estrazione da testo (cerca una colonna 'description')
        else:
            cols_lower = {c.lower(): c for c in df_train.columns}
            description_col = cols_lower.get('description')

            if description_col:
                print(f"--> Estrazione keyword da '{description_col}'...")

                # Funzione helper per cercare le keyword nel testo
                def extract_keywords(text):
                    if not isinstance(text, str): return np.nan
                    text_lower = text.lower()
                    # Trova quali descrittori sono presenti nella descrizione
                    found = [d for d in descriptors if d.lower() in text_lower]
                    if not found: return np.nan # Se nessuna keyword è trovata, la riga sarà scartata
                    return " - ".join(found)

                # Applica l'estrazione
                df_train[actual_target_col] = df_train[description_col].apply(extract_keywords)
                
                # Cerca la colonna description anche nel test set
                if description_col in df_test.columns: 
                    test_desc_col = [c for c in df_test.columns if c.lower() == 'description'][0]
                    df_test[actual_target_col] = df_test[test_desc_col].apply(extract_keywords)
                else:
                    df_test[actual_target_col] = np.nan 

                print(f"--> Target generato. Righe non pertinenti verranno rimosse.")
            else:
                 raise ValueError(f"I descrittori {descriptors} non sono colonne e manca la colonna 'description'.")
    else:
        # Nessun descrittore speciale, usa la colonna target standard fornita
        actual_target_col = target_col

    # --- RIMOZIONE NaN ---
    # Il KNN non tollera valori mancanti. Rimuoviamo le righe con NaN nelle feature o nel target.
    cols_needed_train = feature_cols + [actual_target_col]
    
    # Verifica che le colonne esistano
    if not set(cols_needed_train).issubset(df_train.columns):
         missing = set(cols_needed_train) - set(df_train.columns)
         raise ValueError(f"Colonne mancanti: {missing}")

    initial_len = len(df_train)
    df_train.dropna(subset=cols_needed_train, inplace=True)
    
    if len(df_train) < initial_len:
        print(f"⚠️  ATTENZIONE: Rimosse {initial_len - len(df_train)} su {initial_len} righe (NaN o keyword assenti).")
        if len(df_train) == 0: raise ValueError("Errore: Dataset Training vuoto.")

    # Sul test set rimuoviamo NaN solo dalle feature (il target potrebbe non esserci o servire solo per validazione)
    df_test.dropna(subset=feature_cols, inplace=True)

    # --- Preparazione X e y ---
    X_train = df_train[feature_cols]
    y_train = df_train[actual_target_col]
    X_test = df_test[feature_cols]
    
    # Scaling Z-Score (StandardScaler)
    # Fondamentale per KNN: uniforma le scale per evitare che feature con valori grandi dominino la distanza.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # IMPORTANTE: Usare lo stesso scaler del train!
    
    # Addestramento
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)
    
    # Predizione
    predictions = knn_model.predict(X_test_scaled)
    df_test['prediction'] = predictions
    
    # ==============================================================================
    # --- C. ANALISI BASELINE E INCERTEZZA ---
    # ==============================================================================
    if check_baseline:
        print(f"\n--- 📊 ANALISI COMPARATIVA E INCERTEZZA (k={k}) ---")
        
        # Cross-Validation stratificata (mantiene le proporzioni delle classi)
        n_splits = 10
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        try:
            # 1. KNN Reale: Performance media del modello sui dati corretti
            real_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            r_mean, r_std = real_scores.mean(), real_scores.std()
            
            # 2. Baseline (Dummy): Performance di un modello che sceglie sempre la classe più frequente
            dummy = DummyClassifier(strategy="most_frequent")
            dummy_scores = cross_val_score(dummy, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            d_mean, d_std = dummy_scores.mean(), dummy_scores.std()
            
            # 3. Random Permutation: Performance addestrando su etichette mescolate a caso
            # (Serve a capire se il modello impara pattern reali o rumore)
            y_random = np.random.permutation(y_train)
            random_scores = cross_val_score(knn_model, X_train_scaled, y_random, cv=cv, scoring='accuracy')
            rnd_mean, rnd_std = random_scores.mean(), random_scores.std()

            print(f"Target: {actual_target_col}")
            print(f"{'Modello':<25} | {'Accuracy Media':<10} | {'Incertezza (dev_std)':<15}")
            print("-" * 60)
            print(f"1. KNN Reale (CV)         | {r_mean:.4f}      | ± {r_std:.4f}")
            print(f"2. Baseline (Maggioranza) | {d_mean:.4f}      | ± {d_std:.4f}")
            print(f"3. Random (Caso puro)     | {rnd_mean:.4f}      | ± {rnd_std:.4f}")
            print("-" * 60)
            
            # Analisi robustezza: Confronto intervalli di confidenza (approx 2 deviazioni standard)
            knn_lower_bound = r_mean - (2 * r_std)
            baseline_upper_bound = d_mean + (2 * d_std)
            
            print(f"\n🔍 Analisi Statistica:")
            if knn_lower_bound > baseline_upper_bound:
                print(f"✅ RISULTATO ROBUSTO: L'intervallo del modello è interamente sopra la baseline.")
            elif r_mean > d_mean:
                print(f"⚠️ RISULTATO INCERTO: Media migliore, ma intervalli sovrapposti (poca significatività).")
            else:
                print(f"❌ FALLIMENTO: Il modello non batte la baseline.")

        except Exception as e:
            print(f"⚠️ Errore nel calcolo baseline/incertezza: {e}")
        print("-" * 50)

    # --- A. GESTIONE METRICHE (Test Set) ---
    if print_metrics:
        # Calcoliamo le metriche solo se abbiamo i target reali nel test set
        if actual_target_col in df_test.columns:
            mask_valid = df_test[actual_target_col].notna()
            y_true_clean = df_test.loc[mask_valid, actual_target_col]
            predictions_clean = df_test.loc[mask_valid, 'prediction']
            
            if len(y_true_clean) > 0:
                acc = accuracy_score(y_true_clean, predictions_clean)
                prec = precision_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                rec = recall_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                f1 = f1_score(y_true_clean, predictions_clean, average='weighted', zero_division=0)
                
                print(f"\n--- Performance KNN(k={k}) su Test Set (Dati mai visti) ---")
                print(f"Target:    {actual_target_col}")
                print(f"Accuracy:  {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall:    {rec:.4f}")
                print(f"F-Measure: {f1:.4f}")
                print("---------------------------------------")
            else:
                 print("ATTENZIONE: Test Set privo di target validi.")

    # --- B. GESTIONE PLOT ---
    if make_plot:
        desc_name = make_safe_descriptor_name(descriptors) if descriptors else str(target_col)
        
        # 1. Costruzione percorso file sicuro
        base_dir = "figures/classification"
        sub_dir_features = "_".join(feature_cols)
        sub_dir_target = str(target_col)
        
        # Percorso completo: figures/classification/FeatureA_FeatureB/Target/KNN_Nome.png
        output_path = os.path.join(base_dir, sub_dir_features, sub_dir_target, f"KNN_{desc_name}.png")

        # 2. Creazione Directory (CORREZIONE IMPORTANTE)
        # os.makedirs con exist_ok=True crea ricorsivamente tutte le cartelle mancanti
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True) 

        # 3. Generazione Grafico
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_train,
            x=feature_cols[0], 
            y=feature_cols[1], 
            hue=actual_target_col, 
            palette='viridis',
            s=100,
            edgecolor='black',
            alpha=0.8
        )
        plt.title(f'Training Data: Classi Estratte\nTarget: {actual_target_col}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 4. Salvataggio e chiusura
        plt.savefig(output_path) 
        plt.close() # Libera memoria

    return df_test
