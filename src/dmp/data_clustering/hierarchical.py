from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmp.config import VERBOSE

def hierarchical_clustering(x_column, y_column, n_clusters=3, linkage='ward', **kwargs):
    """Applica l'algoritmo di clustering gerarchico ai dati e produce uno scatter plot dei cluster.

    Input:
        x_column: lista o colonna dei dati corrispondente alla coordinata x
        y_column: lista o colonna dei dati corrispondente alla coordinata y
        n_clusters: numero di cluster da formare
        linkage: tipo di collegamento ('ward', 'complete', 'average', 'single')
        title, x_label, y_label: etichette opzionali per il grafico

    Output:
        fig: figura matplotlib con i cluster
        labels: etichette assegnate ai punti
    """

    # Pulizia dei dati: ignoro punti con NaN
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    # Creo un array (N, 2)
    x_zipped = np.column_stack((x_data, y_data))

    # Applico il clustering gerarchico
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(x_zipped)

    # Plot dei risultati
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", n_clusters)

    for label in range(n_clusters):
        color = palette[label % len(palette)]
        ax.scatter(
            x_zipped[labels == label, 0],
            x_zipped[labels == label, 1],
            s=50, c=[color], alpha=0.6, label=f'Cluster {label}', edgecolors='none'
        )

    ax.set_title(kwargs.get("title", f"Hierarchical Clustering ({linkage}, n={n_clusters})"))
    ax.set_xlabel(kwargs.get("x_label", "X"))
    ax.set_ylabel(kwargs.get("y_label", "Y"))
    ax.grid(True)
    ax.legend()

    if VERBOSE:
        print(f"[INFO] Hierarchical clustering completato con {n_clusters} cluster (linkage='{linkage}')")

    return fig, labels

def plot_dendrogram(x_column, y_column, linkage_method='ward', truncate_mode=None, p=30, **kwargs):
    """Visualizza il dendrogramma per il clustering gerarchico.

    Input:
        x_column, y_column: liste o colonne con le coordinate
        linkage_method: tipo di collegamento ('ward', 'complete', 'average', 'single')
        truncate_mode: se 'lastp', mostra solo gli ultimi p cluster fusi
        p: numero di cluster da mostrare se truncate_mode è impostato
        title, x_label, y_label: etichette opzionali per il grafico

    Output:
        fig: figura matplotlib del dendrogramma
    """

    # Pulizia dei dati
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    x_zipped = np.column_stack((x_data, y_data))

    # Calcolo la matrice di linkage
    Z = linkage(x_zipped, method=linkage_method)

    # Plot del dendrogramma
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        truncate_mode=truncate_mode,  # ad esempio 'lastp' per mostrare solo una parte
        p=p,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        ax=ax
    )

    ax.set_title(kwargs.get("title", f"Dendrogramma ({linkage_method})"))
    ax.set_xlabel(kwargs.get("x_label", "Campioni"))
    ax.set_ylabel(kwargs.get("y_label", "Distanza di fusione"))
    ax.grid(True)

    if VERBOSE:
        print(f"[INFO] Dendrogramma generato con metodo '{linkage_method}'")

    return fig
