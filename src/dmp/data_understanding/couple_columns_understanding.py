import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from dmp.utils import save_figure, filter_column

def generate_scatterplots(df, columns=None, output_dir="figures/scatterplots", title="Scatterplot Matrix", filter_outliers=(0.05,0.95)):
    """
    Genera una griglia di scatterplot per tutte le coppie uniche di colonne numeriche.
    (Non ripete coppie simmetriche e non include la diagonale.)

    Parameters
    ----------
    df : pandas.DataFrame
        Il DataFrame di input.
    columns : list[str] o None
        Colonne da includere. Se None, usa tutte le colonne numeriche.
    output_dir : str, default="figures/scatterplots"
        Directory dove salvare l'immagine.
    title : str
        Titolo della figura.
    filter_outliers : tuple(float, float)
        Percentili per filtrare i dati (default: (0.05, 0.95)).

    Output
    ------
    Nessun ritorno. Salva un file PNG con la griglia di scatterplot.
    """

     # Filtra outliers se specificato
    if filter_outliers:
        df = filter_column(df, columns, by_percentile=True, percentiles=filter_outliers)
    else:
        pass

    # Se columns non è specificato, usa tutte le colonne numeriche
    if columns is None:
        df_numeric = df.select_dtypes(include=["number"])
    else:
        df_numeric = df[columns].select_dtypes(include=["number"])

    numeric_cols = df_numeric.columns.tolist()

    if len(numeric_cols) < 2:
        print("⚠️ Servono almeno due colonne numeriche per generare scatterplot.")
        return

    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)

    # Crea tutte le coppie uniche di colonne
    pairs = list(itertools.combinations(numeric_cols, 2))

    ncols = 9
    nrows = math.ceil(len(pairs) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i, (x_col, y_col) in enumerate(pairs):
        axes[i].scatter(df_numeric[x_col], df_numeric[y_col], alpha=0.6)
        axes[i].set_xlabel(x_col)
        axes[i].set_ylabel(y_col)
        axes[i].set_title(f"{x_col} vs {y_col}")

    # Rimuovi assi vuoti se la griglia non è piena
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Salva un’unica immagine
    if filter_outliers:
        file_path = os.path.join(output_dir, "scatterplot_matrix_cleaned.png")
    else:
        file_path = os.path.join(output_dir, "scatterplot_matrix_unique.png")
    plt.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"✅ Scatterplot matrix salvata in: {file_path}")


def generate_correlation_heatmap(df, columns=None, output_dir="figures/heatmaps", title="Matrice_di_correlazione"):
    """
    Genera una heatmap della matrice di covarianza per le colonne specificate (o tutte le numeriche).

    Parameters
    ----------
    df : pandas.DataFrame
        Il DataFrame di input.
    columns : list[str] o None
        Colonne da includere nella matrice di correlazione.
        Se None, vengono usate tutte le colonne numeriche.
    output_dir : str, default="figures/heatmaps"
        Directory dove salvare l'immagine.
    title : str
        Titolo del grafico.

    Output
    ------
    Nessun ritorno. Salva la heatmap come file PNG nella directory specificata.
    """

    # Se columns non è specificato, usa tutte le colonne numeriche
    if columns is None:
        df_numeric = df.select_dtypes(include=["number"])
    else:
        df_numeric = df[columns].select_dtypes(include=["number"])

    numeric_cols = df_numeric.columns.tolist()

    if len(numeric_cols) < 2:
        print("⚠️ Servono almeno due colonne numeriche per calcolare la covarianza.")
        return

    # Calcola la matrice di correlazione
    cor_matrix = df_numeric.corr()

    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)

    # Crea la figura
    plt.figure(figsize=(8, 6))
    plt.imshow(cor_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar(label="Covarianza")
    plt.title(title)
    plt.xticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols)
    plt.tight_layout()

    # Salva il file
    file_path = save_figure(plt, title, folder=output_dir, extension=".png")

    print(f"✅ Heatmap di covarianza salvata in: {file_path}")