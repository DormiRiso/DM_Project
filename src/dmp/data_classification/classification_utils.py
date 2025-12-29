import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Scikit-Learn: Preprocessing e Composizione
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, label_binarize
from sklearn.compose import ColumnTransformer

# Scikit-Learn: Modelli e Validazione
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Scikit-Learn: Metriche e Visualizzazione Performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay, precision_recall_curve
)

# Import locali/personalizzati
from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name

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
        transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols))

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
    print(f"\n--- 📊 ANALISI STATISTICA BASELINE ({model_name}) ---")
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    n = len(y)  # Numero totale di campioni
    z_alpha_2 = norm.ppf(1 - 0.05/2) # Valore critico Z per alpha=0.05 (circa 1.96)

    def calculate_significance(acc1, acc2, n1, n2):
        """Calcola l'intervallo di confidenza per la differenza tra due proporzioni."""
        d_hat = acc1 - acc2
        # Calcolo della varianza della differenza (come da tua formula)
        # Nota: usiamo la radice quadrata per ottenere la deviazione standard (sigma_d)
        var_d = (acc1 * (1 - acc1) / n1) + (acc2 * (1 - acc2) / n2)
        sigma_d = np.sqrt(var_d)
        
        margin_error = z_alpha_2 * sigma_d
        
        return d_hat, margin_error

    try:
        # 1. Performance Modello Reale
        real_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        r_mean = real_scores.mean()
        
        # 2. Baseline Dummy (Most Frequent)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring='accuracy')
        d_mean = dummy_scores.mean()
        
        # 3. Random Permutation
        y_rnd = np.random.permutation(y)
        rnd_scores = cross_val_score(model, X, y_rnd, cv=cv, scoring='accuracy')
        rnd_mean = rnd_scores.mean()

        print(f"{'Confronto':<25} | {'Diff (d̂)':<10} | {'Intervallo di Confidenza (95%)':<30}")
        print("-" * 80)

        comparisons = [
            ("Reale vs Dummy", r_mean, d_mean),
            ("Reale vs Random", r_mean, rnd_mean)
        ]

        for label, acc1, acc2 in comparisons:
            d_hat, margin_error = calculate_significance(acc1, acc2, n, n)
            status = "✅ SIGNIFICATIVO" if d_hat - margin_error > 0 else "❌ NON SIGNIFICATIVO"
            
            print(f"{label:<25} | {d_hat:.4f}  +/- {margin_error:.4f} -> {status}")

    except Exception as e:
        print(f"⚠️ Errore durante l'analisi: {e}")
    print("-" * 80)    
    


# --- FUNZIONE INTERNA PER GESTIRE I PERCORSI DI SALVATAGGIO ---
def _get_save_path(model_tag, feature_names, descriptors, target_name, filename, base_dir="figures/classification", subfolder=None):
    """
    Costruisce il percorso di salvataggio.
    Se 'subfolder' è specificato, crea una sottocartella (es. ROC_Curves).
    """
    
    # 1. Feature string
    if model_tag == "NB":
        safe_feat_str = ""
    else:
        feats_joined = "_".join(feature_names[:3])
        suffix = "_etc" if len(feature_names) > 3 else ""
        safe_feat_str = f"{feats_joined}{suffix}"
    
    # 2. Descriptor string
    try:
        from dmp.data_understanding.analysis_by_descriptors import make_safe_descriptor_name
        desc_name = make_safe_descriptor_name(descriptors) if descriptors else str(target_name)
    except ImportError:
        desc_name = "_".join(descriptors) if isinstance(descriptors, list) else str(target_name)

    # 3. COSTRUZIONE PATH con SOTTOCARTELLA
    # Struttura: figures/classification / KNN / FeatureName / [Sottocartella Opzionale]
    out_dir = os.path.join(base_dir, str(model_tag), safe_feat_str)
    
    if subfolder:
        out_dir = os.path.join(out_dir, subfolder)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 4. Nome file finale
    final_name = f"{filename}_{desc_name}.png"
    
    return os.path.join(out_dir, final_name)

def _plot_knn_k_search(X_train, y_train, model_tag, feature_names, descriptors, target_name, max_k=300, step_k=15, cv=10):
    """
    Esegue Cross-Validation per trovare K ottimale e SALVA il grafico.
    Adatta dinamicamente il range e lo step se i dati sono pochi.
    """
    
    n_samples = len(X_train)
    min_k = 1 
    
    # 1. Calcola il vero limite superiore per k (non può superare i campioni - 1)
    #    Se n_samples è molto piccolo (es. 20), real_max_k sarà 19.
    real_max_k = min(max_k, len(X_train) - 1)
    k_range = range(1, real_max_k + 1, step_k)
    
    # ... (loop di cross validation e plot invariato) ...
    k_scores = []
    for k in k_range:
        try:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn_temp, X_train, y_train, cv=cv, scoring='accuracy')
            k_scores.append(scores.mean())
        except Exception:
            k_scores.append(0)

    plt.figure(figsize=(12, 6))
    plt.plot(k_range, k_scores, marker='o', linestyle='-', color='purple')
    plt.title(f'Accuratezza media (CV) al variare di K (target : {target_name})')
    plt.xlabel('Valore di K')
    plt.ylabel(f'Cross-Validated Accuracy [cv={cv}]')
    ticks = list(k_range)
    plt.xticks(ticks, rotation=45 if len(ticks) > 15 else 0)
    plt.grid(True, alpha=0.3)
    
    # --- CORREZIONE QUI SOTTO ---
    fname = f"{model_tag}_K_Search_Accuracy"
    # Aggiunto model_tag
    out_path = _get_save_path(model_tag, feature_names, descriptors, target_name, fname)
    # ----------------------------
    
    plt.savefig(out_path)
    plt.close()
   
def _plot_separated_roc(model, X_test, y_test, model_tag, feature_names, descriptors, target_name):
    """
    Genera e SALVA un unico plot contenente le curve ROC per tutte le classi (One-vs-Rest).
    """
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        print("⚠️ Il modello non supporta predict_proba. Impossibile salvare ROC.")
        return

    classes = model.classes_
    n_classes = len(classes)
    
    # Binarizzazione (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Fix per caso binario puro (se label_binarize restituisce una sola colonna)
    if n_classes == 2 and y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    print(f"   💾 Salvataggio Curve ROC Combinate per {model_tag}...")

    # Creazione della figura UNICA prima del ciclo
    plt.figure(figsize=(10, 8))
    
    # Generazione colori diversi per ogni classe
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        current_class = classes[i]
        
        # Calcolo FPR, TPR e AUC per la classe corrente
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Aggiunta della curva al plot
        plt.plot(fpr, tpr, lw=2, color=colors(i),
                 label=f'Class "{current_class}" (AUC = {roc_auc:.2f})')

    # Linea diagonale (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Configurazione finale del plot unico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class ROC Curve: {model_tag}\n(Target: {target_name})')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(alpha=0.3)
    
    # Costruzione nome file e salvataggio
    fname = f"{model_tag}_Combined_ROC"
    
    out_path = _get_save_path(model_tag, feature_names, descriptors, target_name, fname)
    
    plt.savefig(out_path)
    plt.close()
 
def _plot_confusion_matrix(model, predictions, y_test, model_tag, feature_names, descriptors, target_name):
    print(f"   💾 Salvataggio Confusion Matrix per {model_tag}...")
    
    y_pred = predictions
    
    # --- 1. Calcolo delle Metriche ---
    acc = accuracy_score(y_test, y_pred)
    # 'weighted': calcola la media pesata in base al numero di istanze per classe (utile se sbilanciato)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # --- 2. Creazione Plot ---
    # Aumento leggermente l'altezza (figsize) per far spazio al testo sotto
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    # Plot della matrice
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    
    plt.title(f"Confusion Matrix: {model_tag}\n(Target: {target_name})")
    
    # --- 3. Aggiunta statistiche come footer ---
    stats_text = (
        f"Accuracy: {acc:.2%}\n"
        f"Precision: {prec:.2%}\n"
        f"Recall: {rec:.2%}"
    )
    
    # figtext scrive coordinate relative alla figura intera (0,0 basso-sx, 1,1 alto-dx)
    # y=0.02 posiziona il testo in fondo
    plt.figtext(0.5, 0.02, stats_text, horizontalalignment='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray", alpha=0.8))
    
    # Aggiusta i margini per non tagliare il testo in basso
    plt.subplots_adjust(bottom=0.2)
    
    # --- 4. Salvataggio ---
    fname = f"{model_tag}_Confusion_Matrix"
    out_path = _get_save_path(model_tag, feature_names, descriptors, target_name, fname)
    
    plt.savefig(out_path)
    plt.close()
    
def _plot_separated_precision_recall(model, X_test, y_test, model_tag, feature_names, descriptors, target_name):
    """
    Genera e SALVA un unico plot contenente le curve Precision-Recall per tutte le classi (One-vs-Rest).
    """
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        print("⚠️ Il modello non supporta predict_proba. Impossibile salvare P-R Curve.")
        return

    classes = model.classes_
    n_classes = len(classes)
    
    # Binarizzazione (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Fix per caso binario puro
    if n_classes == 2 and y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    print(f"   💾 Salvataggio Curva Precision-Recall Combinata per {model_tag}...")

    # Creazione della figura UNICA prima del ciclo
    plt.figure(figsize=(10, 8))
    
    # Generazione colori diversi per ogni classe
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        current_class = classes[i]
        
        # Calcolo curve
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_score[:, i])
        
        # Calculate prevalence (baseline)
        prevalence = np.sum(y_test_bin[:, i]) / len(y_test_bin)

        # Aggiunta della curva al plot comune
        plt.plot(recall, precision, lw=2, color=colors(i),
                 label=f'Class "{current_class}" (AP = {avg_precision:.2f})')

        # Baseline che dipende dalla % dei positivi
        plt.axhline(y=prevalence, color=colors(i), linestyle='--', alpha=0.5, 
            label=f'Baseline "{current_class}" ({prevalence:.2f})')
    
    # Configurazione finale del plot unico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Multi-class Precision-Recall Curve: {model_tag}\n(Target: {target_name})')
    plt.legend(loc="lower left", fontsize='small')
    plt.grid(alpha=0.3)
    
    # Costruzione nome file e salvataggio
    fname = f"{model_tag}_Combined_Precision_Recall"
    
    # Assicurati che _get_save_path sia accessibile o importata
    out_path = _get_save_path(model_tag, feature_names, descriptors, target_name, fname)
    
    plt.savefig(out_path)
    plt.close()

def _plot_summary_subplot(model, predictions, X_test, y_test, model_tag, feature_names, descriptors, target_name):
    """
    Creates a 1x3 subplot containing the Confusion Matrix, ROC Curves, and Precision-Recall Curves.
    """
    try:
        y_score = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    except AttributeError:
        print(f"⚠️ Model {model_tag} does not support probability estimates.")
        return

    classes = model.classes_
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    if n_classes == 2 and y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    # Create figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.get_cmap('tab10', n_classes)

    # --- 1. Confusion Matrix (Left) ---
    cm = confusion_matrix(y_test, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion Matrix\nAccuracy: {accuracy_score(y_test, predictions):.2%}")

    # --- 2. ROC Curves (Center) ---
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, lw=2, color=colors(i), label=f'Class {classes[i]} ({roc_auc:.2f})')
    
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_title(f"Multi-class ROC (Target: {target_name})")
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc="lower right", fontsize='x-small')
    axes[1].grid(alpha=0.3)

    # --- 3. Precision-Recall Curves (Right) ---
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        prevalence = np.sum(y_test_bin[:, i]) / len(y_test_bin)
        
        axes[2].plot(recall, precision, lw=2, color=colors(i), label=f'Class {classes[i]} (AP={ap:.2f})')
        axes[2].axhline(y=prevalence, color=colors(i), linestyle='--', alpha=0.4)

    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].legend(loc="lower left", fontsize='x-small')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    
    # Save using your custom path function
    fname = f"{model_tag}_Full_Summary_Report"
    out_path = _get_save_path(model_tag, feature_names, descriptors, target_name, fname)
    plt.savefig(out_path)
    plt.close()