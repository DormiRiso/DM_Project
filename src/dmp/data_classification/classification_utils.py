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