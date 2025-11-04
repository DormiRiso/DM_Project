import pandas as pd
import matplotlib.pyplot as plt
import os

def make_hist(df, colonna, bins=10, titolo=None):
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
    
    # Crea la cartella plots se non esiste
    os.makedirs("plots", exist_ok=True)
    
    # Crea il grafico
    plt.figure(figsize=(8, 5))
    plt.hist(df[colonna].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    plt.title(titolo if titolo else f"Istogramma di '{colonna}'")
    plt.xlabel(colonna)
    plt.ylabel("Frequenza")
    plt.grid(axis='y', alpha=0.75)
    
    # Salvataggio del file
    nome_file = titolo if titolo else colonna
    nome_file = nome_file.replace(" ", "_").lower() + ".png"
    percorso_file = os.path.join("plots", nome_file)
    plt.savefig(percorso_file, bbox_inches='tight')
    plt.close()
    
    print(f"Istogramma salvato in: {percorso_file}")