from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmp.config import VERBOSE

def dbscan(x_column, y_column, eps=0.3, min_samples=5, **kwargs):
    """Applica l'algoritmo DBSCAN ai dati forniti e produce uno scatter plot dei cluster,
    con box informativo contenente eps, N_min e altre statistiche."""

    # Pulizia dei dati: ignoro punti con NaN
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    if not x_data or not y_data:
        raise ValueError("Le colonne fornite sono vuote o contengono solo valori validi NaN.")

    # Creo un array (N, 2) per DBSCAN
    X = np.column_stack((x_data, y_data))

    # Applico DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels = db.labels_

    # Numero di cluster trovati (escludendo il rumore, che ha label = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    n_points = len(labels)

    # Plot dei risultati
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", max(n_clusters, 1))

    # Colori per cluster (i punti rumorosi saranno in nero)
    for label in set(labels):
        if label == -1:
            color = 'k'
            label_name = 'Rumore'
        else:
            color = palette[label % len(palette)]
            label_name = f'Cluster {label}'
        ax.scatter(
            X[labels == label, 0],
            X[labels == label, 1],
            s=50, c=[color], label=label_name, alpha=0.6, edgecolors='none'
        )

    # Titoli e assi
    ax.set_title(kwargs.get("title", f"DBSCAN Clustering (eps={eps}, N_min={min_samples})"))
    ax.set_xlabel(kwargs.get("x_label", "X"))
    ax.set_ylabel(kwargs.get("y_label", "Y"))
    ax.grid(True)

    # --- BOX INFORMATIVO ---
    info_text = (
        f"Parametri:\n"
        f"  • eps = {eps}\n"
        f"  • N_min = {min_samples}\n\n"
        f"Risultati:\n"
        f"  • Cluster trovati = {n_clusters}\n"
        f"  • Punti totali = {n_points}\n"
        f"  • Rumore = {n_noise}"
    )

    ax.text(
        1.02, 0.5, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray")
    )

    # Legenda spostata di lato per non sovrapporsi al box
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    if VERBOSE:
        print(f"[INFO] DBSCAN ha trovato {n_clusters} cluster "
              f"({n_noise} punti di rumore su {n_points} totali). eps={eps}, N_min={min_samples}")

    return fig, labels
