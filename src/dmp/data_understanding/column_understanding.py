import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dmp.utils import save_figure
from dmp.config import VERBOSE

def plot_column_analysis(df, colonna, bins=30):
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
    # Calcola i percentili che si useranno per i box plot
    percentiles = np.percentile(data, [5, 25, 50, 75, 95])
    
    # Dati puliti
    clean_mask = (data >= percentiles[0]) & (data <= percentiles[4])
    clean_data = data[clean_mask]
    num_outliers = len(data) - len(clean_data)

    # Sx: istogramma originale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    ax1.set_title(f'Istogramma originale di {colonna}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Conteggi")

    ax2=ax1.twinx()
    ax2.set_ylim(0, 1)

    larghezza_boxplot = 0.15
    y_boxplot = 0.75    
    
    ax2.boxplot(data, whis=[5, 95], showfliers=True, 
                vert=False, widths=larghezza_boxplot, positions=[y_boxplot], 
                medianprops=dict(color="red", linewidth=1.5))

    ax2.get_yaxis().set_visible(False)

    # Aggiungi textbox con percentili originali sull'asse del boxplot sinistro
    pct_text_orig = (
        f"Percentili (originale):\n"
        f"5%: {percentiles[0]:.2f}\n"
        f"25%: {percentiles[1]:.2f}\n"
        f"50%: {percentiles[2]:.2f}\n"
        f"75%: {percentiles[3]:.2f}\n"
        f"95%: {percentiles[4]:.2f}"
    )
    ax1.text(0.98, 0.98, pct_text_orig, transform=ax1.transAxes,
             fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    

    # Dx: istogramma pulito
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.hist(clean_data, bins=bins, edgecolor='black', alpha=0.7)
    ax3.set_title(f'Istogramma pulito di {colonna}\n(5-95 percentili)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel("Conteggi")

    ax4 = ax3.twinx()
    ax4.set_ylim(0, 1)
    y_boxplot = 0.75 
    larghezza_boxplot = 0.15


    ax4.boxplot(clean_data, whis=[5, 95], showfliers=True,
                vert=False, widths=larghezza_boxplot, positions=[y_boxplot],
                medianprops=dict(color="red", linewidth=1.5))

    ax4.get_yaxis().set_visible(False)

    # Aggiungi textbox con percentili puliti e conteggio outliers sull'asse destro
    if len(clean_data) > 0:
        percentiles_clean = np.percentile(clean_data, [5, 25, 50, 75, 95])
        pct_text_clean = (
            f"Percentili (pulito):\n"
            f"5%: {percentiles_clean[0]:.2f}\n"
            f"25%: {percentiles_clean[1]:.2f}\n"
            f"50%: {percentiles_clean[2]:.2f}\n"
            f"75%: {percentiles_clean[3]:.2f}\n"
            f"95%: {percentiles_clean[4]:.2f}\n\n"
            f"Outliers rimossi: {num_outliers} ({num_outliers/len(data)*100:.2f}%)"
        )
    else:
        pct_text_clean = f"Nessun dato nel range 5-95%. Outliers rimossi: {num_outliers}"

    ax3.text(0.98, 0.98, pct_text_clean, transform=ax3.transAxes,
             fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))


    plt.suptitle(f'Analisi statistica di {colonna}', fontsize=16, y=1.02)

    # Salva
    file_path = save_figure(plt, f"analisi_singola_{colonna}", "figures/histograms", ".png")
    if VERBOSE:
        print(f"Analysis plot saved in: {file_path}")
    plt.close()

def analizza_colonne_numeriche(df):
    """
    Crea grafici di analisi completi per tutte le colonne numeriche del DataFrame.
    """
    for colonna in df.columns:
        if pd.api.types.is_numeric_dtype(df[colonna]):
            if len(df[colonna].dropna().unique()) > 1:  # Se ci sono almeno due punti diversi...
                # print(f"\nAnalyzing column: {colonna}")
                plot_column_analysis(df, colonna)
            else:
                pass # print(f"\nSkipping {colonna}: insufficient variation in data")
        else:
            pass # print(f"\nSkipping {colonna}: not a numerical column")
