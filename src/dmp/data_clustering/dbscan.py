from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmp.config import VERBOSE


def dbscan(x_column, y_column, eps=0.3, min_samples=5, show_knee=True, **kwargs):
    """
    Esegue DBSCAN e produce un doppio grafico: 
    1. Analisi della K-distance per la scelta di eps.
    2. Scatter plot dei cluster risultanti.
    """

    # --- 1. Preparazione e Pulizia Dati (CORRETTA) ---
    x_data = []
    y_data = []
    
    # Iteriamo sulle coppie per garantire che le lunghezze rimangano uguali
    for x, y in zip(x_column, y_column):
        # Se uno dei due è NaN, saltiamo la coppia
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    # Controllo di sicurezza se dopo la pulizia non rimangono dati
    if not x_data:
        raise ValueError("Dopo la rimozione dei NaN non sono rimasti dati sufficienti.")

    X = np.column_stack((x_data, y_data))

    # --- 2. Calcolo K-Distance (per il plot diagnostico) ---
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    k_distances = np.sort(distances[:, min_samples-1], axis=0)

    # --- 3. Esecuzione DBSCAN ---
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    # --- 4. Plotting (Layout a due colonne) ---
    fig, (ax_knee, ax_scatter) = plt.subplots(1, 2, figsize=(15, 6))
    
    # A. SOTTOPLOT: K-Distance Plot
    ax_knee.plot(k_distances, color='royalblue', linewidth=2)
    ax_knee.axhline(y=eps, color='r', linestyle='--', label=f'Eps scelto ({eps})')
    ax_knee.set_title("Analisi per scelta Epsilon (K-dist)")
    ax_knee.set_xlabel("Punti ordinati")
    ax_knee.set_ylabel(f"Distanza dal {min_samples}° vicino")
    ax_knee.legend()
    ax_knee.grid(True)

    # B. SOTTOPLOT: Scatter Plot
    palette = sns.color_palette("husl", max(n_clusters, 1))
    # Ottimizzazione: plotta solo le label uniche trovate
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            color = 'k'
            label_name = 'Rumore'
            alpha = 0.3 # Rendiamo il rumore meno invasivo visivamente
        else:
            color = palette[label % len(palette)]
            label_name = f'Cluster {label}'
            alpha = 0.6
            
        mask = (labels == label)
        ax_scatter.scatter(X[mask, 0], X[mask, 1], s=40, c=[color], 
                           label=label_name, alpha=alpha, edgecolors='none')

    ax_scatter.set_title(kwargs.get("title", "Risultato DBSCAN"))
    ax_scatter.set_xlabel(kwargs.get("x_label", "X"))
    ax_scatter.set_ylabel(kwargs.get("y_label", "Y"))
    ax_scatter.grid(True, alpha=0.3)

    # --- BOX INFORMATIVO ---
    info_text = (
        f"Parametri:\n  • eps = {eps}\n  • N_min = {min_samples}\n\n"
        f"Risultati:\n  • Cluster = {n_clusters}\n  • Rumore = {n_noise}\n"
        f"  • Punti totali = {len(X)}"
    )
    # Posiziona il box informativo leggermente fuori dal plot
    ax_scatter.text(1.05, 0.5, info_text, transform=ax_scatter.transAxes,
                    fontsize=10, va='center', bbox=dict(boxstyle="round", facecolor="whitesmoke"))

    # Legenda esterna
    ax_scatter.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()

    if VERBOSE:
        print(f"[INFO] DBSCAN completato: {n_clusters} cluster trovati su {len(X)} punti.")

    return fig, labels
