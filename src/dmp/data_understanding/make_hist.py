import os
import pandas as pd
import matplotlib.pyplot as plt
from dmp.utils import save_figure
from dmp.config import VERBOSE

def make_hist(df, colonna, bins=10, folder = "figures",titolo=None):
    """
    Crea e salva un istogramma di una colonna di un DataFrame Pandas nella cartella 'plots/'.

    Parametri:
    - df (pd.DataFrame): il DataFrame da cui prendere i dati
    - colonna (str): il nome della colonna da visualizzare
    - bins (int): numero di intervalli (default: 10)
    - titolo (str): titolo del grafico (opzionale)
    """
    if colonna not in df.columns:
        raise ValueError(f"La colonna '{colonna}' non esiste nel DataFrame.")

    # Crea il grafico
    plt.figure(figsize=(8, 5))
    data = plt.hist(df[colonna].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    plt.title(titolo if titolo else f"Istogramma di '{colonna}'")
    plt.xlabel(colonna)
    plt.ylabel("Frequenza")
    plt.grid(axis='y', alpha=0.75)
    
    file_path = save_figure(plt, titolo if titolo else colonna, folder = folder, extension=".png")
    
    if VERBOSE:
        print(f"Istogramma salvato in: {file_path}")
    return data