import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dmp.utils import save_figure
from dmp.config import VERBOSE
from dmp.my_graphs import histo_box

def plot_column_analysis(df, df_filtered, colonna, output_path, bins=30):
    """
    Crea un grafico di analisi completo per una colonna numerica con:
    - Istogramma originale (in alto a sinistra)
    - Box plot con i percentili 5,25,50,75,95 (in basso a sinistra)
    - Istogramma pulito senza outliers (in alto a destra)
    - Box plot pulito senza outliers (in basso a destra)
    """
    # Crea subplot 1x2
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    # Prendi dati e rimuovi Nan
    data = df[colonna].dropna()
    # Dati puliti
    clean_data = df_filtered[colonna].dropna()

    num_outliers = len(data) - len(clean_data)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])


    histo_box(ax1, data, colonna, summary=True)
    histo_box(ax2, clean_data, colonna, summary=True, extra_info=f"Outliers rimossi: {num_outliers} ({num_outliers/len(data)*100:.2f}%)")

    # Salva
    file_path = save_figure(plt, f"analisi_singola_{colonna}", output_path, ".png")
    if VERBOSE:
        print(f"Analysis plot saved in: {file_path}")
    plt.close()

def analizza_colonne_numeriche(df, df_filtered, output_path, columns=None):
    """
    Crea grafici di analisi completi per le colonne numeriche specificate del DataFrame.

    Parametri:
        df (pd.DataFrame): Il DataFrame da analizzare.
        columns (list[str] | None): Lista di colonne da analizzare.
            Se None, analizza tutte le colonne numeriche.
    """
    # Se non passo nessuna lista → prendo tutte le colonne numeriche
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    for col in columns:
        if col not in df.columns:
            print(f"⚠️ Colonna '{col}' non trovata nel DataFrame, salto.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠️ Colonna '{col}' non è numerica, salto.")
            continue

        if len(df[col].dropna().unique()) <= 1:
            print(f"⚠️ Colonna '{col}' ha variazione insufficiente, salto.")
            continue

        # Se passa tutti i controlli → analizza la colonna
        plot_column_analysis(df, df_filtered, col, output_path)

