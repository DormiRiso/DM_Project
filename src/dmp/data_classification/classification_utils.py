import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re # Necessario per la sanificazione dei nomi

from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def generate_plots(model, X_train, y_train, X_test, y_test, feature_names, target_name, descriptors, model_tag="Model"):
    """
    Funzione di reportistica visuale che genera una Dashboard di performance in un'unica immagine.
    Include Scatterplot (distribuzione training), Confusion Matrix e ROC Curve (valutazione test).

    Args:
        model: Il modello addestrato (KNN o Naive Bayes).
        X_train, y_train (np.array, pd.Series): Feature e Target del Training (trasformati).
        X_test, y_test (np.array, pd.Series): Feature e Target del Test (trasformati e puliti).
        feature_names (list): Nomi delle feature dopo il preprocessing (es. 'Age_scaled', 'City_London').
        target_name (str): Nome della colonna target.
        descriptors (list): I descrittori usati, per la nomenclatura del file.
        model_tag (str): Prefisso del modello (es. 'NB', 'KNN_3').

    Returns:
        Stampa il percorso del file salvato.
    """
    # --- PREPARAZIONE PERCORSI E FILE NAME ---
    desc_name = make_safe_descriptor_name(descriptors) if descriptors else str(target_name)
    
    # Crea un nome di cartella corto e sicuro basato sul numero totale di feature (post-OHE)
    # per evitare path troppo lunghi o caratteri non validi.
    safe_feat_str = "_".join(feature_names[:3]) + ("_etc" if len(feature_names)>3 else "")
    
    base_dir = "figures/classification"
    # Costruisce il percorso finale, includendo il modello e il nome del target.
    out_path = os.path.join(base_dir, safe_feat_str, f"{model_tag}_Dashboard_{desc_name}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    # --- SETUP GRAFICO (1 riga, 3 colonne) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- 1. SCATTERPLOT (PANNELLO SINISTRO) ---
    ax_scatter = axes[0]
    if X_train.shape[1] >= 2: # Controlla che ci siano almeno 2 feature per il plot 2D
        try:
            # Plotta i dati di training (X_train) usando le prime due dimensioni.
            # L'hue (colore) è dato dalla classe (y_train).
            sns.scatterplot(
                x=X_train[:, 0], y=X_train[:, 1], hue=y_train, 
                palette='tab10', s=50, edgecolor='black', alpha=0.7, ax=ax_scatter
            )
            ax_scatter.set_title(f'Training Data (Projected)')
            # Etichetta gli assi con i nomi delle feature trasformate (feature_names[0], [1]).
            ax_scatter.set_xlabel(feature_names[0])
            ax_scatter.set_ylabel(feature_names[1])
            ax_scatter.legend(loc='upper right', title="Class", fontsize='small')
        except Exception as e:
            ax_scatter.text(0.5, 0.5, f"Scatterplot Error:\n{str(e)}", ha='center', va='center')
    else:
        # Messaggio di fallback se il plot 2D non è possibile
        ax_scatter.text(0.5, 0.5, "Need >= 2 features\nfor Scatterplot", ha='center', va='center')
        ax_scatter.axis('off')

    # --- 2. CONFUSION MATRIX (PANNELLO CENTRALE) ---
    ax_cm = axes[1]
    try:
        # Utilizza il modello per predire su X_test e confrontare con y_test,
        # visualizzando gli errori e i successi (veri/falsi positivi/negativi).
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm, cmap='Blues', colorbar=False)
        ax_cm.set_title(f'Confusion Matrix\n(Test Set)')
    except Exception as e:
        ax_cm.text(0.5, 0.5, f"CM Error:\n{str(e)}", ha='center', va='center')

    # --- 3. ROC CURVE (PANNELLO DESTRO) ---
    ax_roc = axes[2]
    try:
        # Plotta la curva ROC (True Positive Rate vs False Positive Rate).
        # Funziona ottimamente per la classificazione binaria.
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=model_tag)
        ax_roc.plot([0, 1], [0, 1], 'r--', label='Chance') # Aggiunge la linea di classificazione casuale
        ax_roc.set_title('ROC Curve')
    except ValueError:
        # GESTIONE ECCEZIONE: Se il target ha più di due classi (Multiclass), la ROC standard fallisce.
        ax_roc.text(0.5, 0.5, "ROC N/A\n(Multiclass > 2)", ha='center', va='center', fontsize=12)
        ax_roc.set_title('ROC Curve (N/A)')
        ax_roc.axis('off')
    except Exception as e:
        ax_roc.text(0.5, 0.5, f"ROC Error:\n{str(e)}", ha='center', va='center')
        ax_roc.axis('off')

    # --- FINALIZZAZIONE E SALVATAGGIO ---
    plt.suptitle(f"{model_tag} Performance | Target: {target_name}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"Dashboard salvata: {out_path}")
    

# Assumiamo questa funzione helper per i nomi sicuri (come nel tuo codice)
try:
    from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name
except ImportError:
    # Fallback se la libreria non è disponibile
    make_safe_descriptor_name = lambda x: "_".join(x) if isinstance(x, list) else str(x)

def prepare_target_column(df, descriptors, target_col):
    """
    Logica centralizzata per determinare o generare la colonna target.

    Se 'descriptors' è fornito, tenta di creare una colonna target unendo i valori
    delle colonne specificate o estraendo le keyword da una colonna 'description'.

    Args:
        df (pd.DataFrame): DataFrame.
        descriptors (list): Lista di colonne o keyword.
        target_col (str): Nome della colonna target di fallback.

    Returns:
        tuple: (DataFrame con target potenziato, nome effettivo della colonna target)
    """
    df_out = df.copy()
    
    # Caso 1: Descrittori (Target Composto)
    if descriptors and isinstance(descriptors, list) and len(descriptors) > 0:
        actual_target_col = " - ".join(descriptors)
        
        # A. Descrittori come colonne esistenti
        if set(descriptors).issubset(df_out.columns):
            df_out[actual_target_col] = df_out[descriptors].astype(str).agg(' - '.join, axis=1)
            return df_out, actual_target_col
            
        # B. Estrazione da Testo ('description')
        cols_lower = {c.lower(): c for c in df_out.columns}
        description_col = cols_lower.get('description')
        
        if description_col:
            def extract_keywords(text):
                if not isinstance(text, str): return np.nan
                text_lower = text.lower()
                # Trova quali descrittori sono presenti nel testo
                found = [d for d in descriptors if d.lower() in text_lower]
                # Se ne trova, crea l'etichetta, altrimenti NaN (che verrà scartato dopo)
                return " - ".join(found) if found else np.nan

            df_out[actual_target_col] = df_out[description_col].apply(extract_keywords)
            return df_out, actual_target_col
        else:
            # Se descriptors è dato ma la logica fallisce (es. colonna description mancante)
            return df_out, actual_target_col

    # Caso 2: Target Standard
    return df_out, target_col

def clean_and_process_data(train_df, test_df, numeric_cols, categorical_cols, target_col):
    """
    Gestisce NaN, applica Scaling alle numeriche e OneHotEncoding alle categoriche.
    Restituisce matrici numpy pronte per il modello e i nomi delle feature trasformate.
    """
    
    # Definiamo la funzione di pulizia robusta. Se l'input è None, usa lista vuota.
    clean_list = lambda cols: sorted(list(set([c for c in (cols if cols is not None else []) if c and str(c).strip()])))

    # --- 0. SANITIZZAZIONE INPUT ---
    
    # Applichiamo la pulizia, gestendo implicitamente il caso None
    numeric_cols = clean_list(numeric_cols)
    categorical_cols = clean_list(categorical_cols)
    
    # 1. Pulizia: Rimuoviamo None, stringhe vuote ("") e duplicati
    clean_list = lambda cols: sorted(list(set([c for c in cols if c and str(c).strip()])))
    numeric_cols = clean_list(numeric_cols)
    categorical_cols = clean_list(categorical_cols)

    # 2. Controlliamo che una colonna non sia sia numerica che categorica
    intersect = set(numeric_cols).intersection(categorical_cols)
    if intersect:
        print(f"⚠️ ATTENZIONE: Le colonne {intersect} erano sia in numeriche che categoriche. -> Trattate come NUMERICHE.")
        categorical_cols = [c for c in categorical_cols if c not in intersect]

    # 3. Assicuriamoci che il target non sia tra le feature
    if target_col in numeric_cols: numeric_cols.remove(target_col)
    if target_col in categorical_cols: categorical_cols.remove(target_col)

    all_features = numeric_cols + categorical_cols
    
    if not all_features:
        raise ValueError("ERRORE: Nessuna feature valida fornita (le liste erano vuote o contenevano solo stringhe vuote).")

    # 4. CONTROLLO COLONNE DUPLICATE NEL DATAFRAME
    if train_df.columns.has_duplicates:
        dupes = train_df.columns[train_df.columns.duplicated()].tolist()
        raise ValueError(f"ERRORE CRITICO: Il DataFrame di training ha colonne duplicate: {dupes}.")

    cols_needed = list(set(all_features + [target_col]))
    
    # --- 1. Pulizia Training (Rimuove NaN in features/target) ---
    initial_train = len(train_df)
    missing_cols = [c for c in cols_needed if c not in train_df.columns]
    if missing_cols: raise ValueError(f"Colonne mancanti nel Train: {missing_cols}")

    df_train_clean = train_df.dropna(subset=cols_needed).copy()
    dropped_train = initial_train - len(df_train_clean)
    
    if len(df_train_clean) == 0: raise ValueError(f"ERRORE: Training Set vuoto dopo pulizia!")

    # --- 2. Pulizia Test (Rimuove NaN in features/target) ---
    initial_test = len(test_df)
    cols_needed_test = list(set(all_features + [target_col])) if target_col in test_df.columns else all_features
    
    df_test_clean = test_df.dropna(subset=cols_needed_test).copy()
    dropped_test = initial_test - len(df_test_clean)

    # Report
    print("\n--- 🧹 Report Pulizia Dati ---")
    print(f"Training Set: {initial_train} -> {len(df_train_clean)} (Drop: {dropped_train})")
    print(f"Test Set:     {initial_test} -> {len(df_test_clean)} (Drop: {dropped_test})")
    print("------------------------------")

    # --- 3. Separazione X/y ---
    X_train = df_train_clean[all_features]
    y_train = df_train_clean[target_col]
    X_test = df_test_clean[all_features]
    
    # --- 4. Column Transformer (Scaling + Encoding) ---
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols)) # Scaling delle numeriche
    if categorical_cols:
        # OneHotEncoder per categoriche (handle_unknown='ignore' previene crash su nuove categorie nel test)
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
    
    # Fit (calcolo media/std/categorie) su Train
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform (applicazione delle trasformazioni) su Test
    if len(X_test) > 0:
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_test_processed = np.array([])
        
    final_feature_names = preprocessor.get_feature_names_out().tolist()
    
    return X_train_processed, y_train, X_test_processed, df_train_clean, df_test_clean, final_feature_names

def make_metrics(y_true, y_pred, model_name, target_name):
    """ 
    Calcola e stampa le metriche di classificazione standard.
    """
    if len(y_true) == 0: return

    # Calcola le metriche
    acc = accuracy_score(y_true, y_pred)
    # Average='weighted' è usato per metriche su target multi-classe sbilanciati
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Stampa formattata
    print(f"\n--- Performance {model_name} su Test Set ---")
    print(f"Target:    {target_name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F-Measure: {f1:.4f}")
    print("-" * 40)

def run_baseline_analysis(model, X, y, model_name):
    """
    Esegue cross-validation confrontando il Modello Reale con due baseline statistiche:
    1. Dummy Classifier (sceglie la classe più frequente).
    2. Random Permutation (modello allenato su etichette casuali).
    """
    print(f"\n--- 📊 ANALISI BASELINE ({model_name}) ---")
    # Usa Stratified K-Fold per mantenere la proporzione delle classi in ogni fold
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    try:
        # 1. Modello Reale (Performance media sui dati corretti)
        real_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        r_mean, r_std = real_scores.mean(), real_scores.std()
        
        # 2. Baseline Dummy (Modello che sceglie sempre la classe più frequente)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring='accuracy')
        d_mean, d_std = dummy_scores.mean(), dummy_scores.std()
        
        # 3. Random Permutation (Test di robustezza/significatività)
        y_rnd = np.random.permutation(y)
        # Alleniamo il modello reale su etichette casuali
        rnd_scores = cross_val_score(model, X, y_rnd, cv=cv, scoring='accuracy')
        rnd_mean, rnd_std = rnd_scores.mean(), rnd_scores.std()

        # Stampa dei risultati comparativi
        print(f"{'Modello':<25} | {'Acc Media':<10} | {'Std Dev':<15}")
        print("-" * 60)
        print(f"1. {model_name:<22} | {r_mean:.4f}      | ± {r_std:.4f}")
        print(f"2. Baseline (Most Freq)   | {d_mean:.4f}      | ± {d_std:.4f}")
        print(f"3. Random Permutation     | {rnd_mean:.4f}      | ± {rnd_std:.4f}")
        
        # Analisi della significatività (Confronto degli intervalli di confidenza approssimativi)
        lower_bound = r_mean - (2*r_std)
        upper_bound = d_mean + (2*d_std)

        if lower_bound > upper_bound:
            print("\n✅ RISULTATO ROBUSTO: L'intervallo di confidenza del modello supera la baseline (2 std_dev).")
        elif r_mean > d_mean:
            print("\n⚠️ RISULTATO INCERTO: Media migliore, ma gli intervalli si sovrappongono (2 std_dev).")
        else:
            print("\n❌ FALLIMENTO: Il modello non batte la baseline(2 std_dev).")
            
    except Exception as e:
        print(f"⚠️ Errore baseline: {e}")
    print("-" * 50)
