from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dmp.config import VERBOSE


def hierarchical_clustering(x_column, y_column, n_clusters=3, linkage='ward', **kwargs):
    """Applica il clustering gerarchico e mostra uno scatter plot con info aggiuntive."""

    # Pulizia dei dati: ignoro punti con NaN
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    if not x_data or not y_data:
        raise ValueError("Le colonne fornite sono vuote o contengono solo valori NaN.")

    # Creo un array (N, 2)
    X = np.column_stack((x_data, y_data))

    # Applico il clustering gerarchico
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)

    # Plot dei risultati
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", n_clusters)

    for label in range(n_clusters):
        color = palette[label % len(palette)]
        ax.scatter(
            X[labels == label, 0],
            X[labels == label, 1],
            s=50, c=[color], alpha=0.6,
            label=f'Cluster {label}', edgecolors='none'
        )

    # Titoli e assi
    ax.set_title(kwargs.get("title", f"Hierarchical Clustering ({linkage}, n={n_clusters})"))
    ax.set_xlabel(kwargs.get("x_label", "X"))
    ax.set_ylabel(kwargs.get("y_label", "Y"))
    ax.grid(True)

    # --- BOX INFORMATIVO ---
    n_points = len(X)
    info_text = (
        f"Parametri:\n"
        f"  • linkage = {linkage}\n"
        f"  • n_clusters = {n_clusters}\n\n"
        f"Risultati:\n"
        f"  • Punti totali = {n_points}"
    )

    ax.text(
        1.02, 0.5, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray")
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    if VERBOSE:
        print(f"[INFO] Hierarchical clustering completato con {n_clusters} cluster "
              f"(linkage='{linkage}', {len(X)} punti).")

    return fig, labels


def plot_dendrogram(x_column, y_column, linkage_method='ward', truncate_mode=None, p=30, **kwargs):
    """Visualizza il dendrogramma per il clustering gerarchico con informazioni aggiuntive."""

    # Pulizia dei dati
    x_data, y_data = [], []
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(float(x))
        y_data.append(float(y))

    if not x_data or not y_data:
        raise ValueError("Le colonne fornite sono vuote o contengono solo valori NaN.")

    X = np.column_stack((x_data, y_data))

    # Calcolo la matrice di linkage
    Z = linkage(X, method=linkage_method)

    # Plot del dendrogramma
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
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

    # --- BOX INFORMATIVO ---
    info_text = (
        f"Parametri:\n"
        f"  • linkage = {linkage_method}\n"
        f"Risultati:\n"
        f"  • Punti totali = {len(X)}"
    )

    ax.text(
        1.02, 0.5, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray")
    )

    plt.tight_layout()

    if VERBOSE:
        print(f"[INFO] Dendrogramma generato con metodo '{linkage_method}' "
              f"({len(X)} punti, truncate_mode={truncate_mode}, p={p}).")

    return fig
