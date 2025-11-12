from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmp.config import VERBOSE

def dbscan(x_column, y_column, eps=0.3, min_samples=5, **kwargs):
    """Applica l'algoritmo DBSCAN ai dati forniti e produce uno scatter plot dei cluster.

    Input:
        x_column: colonna dei dati corrispondente alla coordinata x
        y_column: colonna dei dati corrispondente alla coordinata y
        eps: raggio massimo per includere un punto nel cluster (default = 0.3)
        min_samples: numero minimo di punti per formare un cluster (default = 5)
        title, x_label, y_label: etichette opzionali per il grafico

    Output:
        fig: figura matplotlib con i cluster individuati
        labels: etichette assegnate a ciascun punto dal DBSCAN
    """

    # Pulizia dei dati: ignoro punti con NaN
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    # Creo un array (N, 2) per DBSCAN
    x_zipped = np.column_stack((x_data, y_data))

    # Applico DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(x_zipped)
    labels = db.labels_

    # Numero di cluster trovati (escludendo il rumore, che ha label = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Plot dei risultati
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", n_clusters)

    # Colori per cluster (i punti rumorosi saranno in nero)
    for label in set(labels):
        if label == -1:
            # Rumore
            color = 'k'
            label_name = 'Noise'
        else:
            color = palette[label % len(palette)]
            label_name = f'Cluster {label}'
        ax.scatter(
            x_zipped[labels == label, 0],
            x_zipped[labels == label, 1],
            s=50, c=[color], label=label_name, alpha=0.6, edgecolors='none'
        )

    ax.set_title(kwargs.get("title", f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})"))
    ax.set_xlabel(kwargs.get("x_label", "X"))
    ax.set_ylabel(kwargs.get("y_label", "Y"))
    ax.grid(True)
    ax.legend()

    if VERBOSE:
        print(f"[INFO] DBSCAN ha trovato {n_clusters} cluster (rumore: {np.sum(labels == -1)} punti)")

    return fig, labels
