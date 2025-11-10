import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dmp.my_graphs import box_plot

def random_sampling(df: pd.DataFrame, n: int, seed: int = None) -> pd.DataFrame:
    """
    Estrae un campione casuale di n righe dal DataFrame.

    Parametri:
        df (pd.DataFrame): Il DataFrame da campionare.
        n (int): Numero di righe da estrarre.
        seed (int, opzionale): Valore per fissare la casualità.

    Ritorna:
        pd.DataFrame: Campione casuale di n righe.
    """
    return df.sample(n=n, random_state=seed)


def distributed_sampling(df: pd.DataFrame, colonna: str, n: int, bins: int = 10, seed: int = None) -> pd.DataFrame:
    np.random.seed(seed)

    #1. Rimuovi righe con NaN nella colonna di riferimento
    df = df.dropna(subset=[colonna]).copy()

    # 2. Discretizza la colonna
    categorie, _ = pd.cut(df[colonna], bins=bins, labels=False, retbins=True, include_lowest=True)

    # 3. Frequenze per bin
    frequenze = pd.Series(categorie).value_counts().sort_index()

    # 4. Calcola le probabilità per ciascun bin (evitando divisioni per zero)
    prob_per_bin = frequenze / frequenze.sum()

    # 5. Mappa le probabilità ai record, e sostituisci i NaN con 0
    pesi = categorie.map(prob_per_bin).fillna(0)

    # 6. Normalizza (in caso di piccoli scarti numerici)
    somma_pesi = pesi.sum()
    if somma_pesi == 0:
        raise ValueError("Tutti i pesi risultano nulli. Controlla la colonna o i bin.")
    pesi = pesi / somma_pesi

    # 7. Campionamento ponderato
    campione_idx = np.random.choice(df.index, size=n, replace=False, p=pesi.values)
    return df.loc[campione_idx].reset_index(drop=True)


def sample_df(df: pd.DataFrame, 
              N: int, 
              method: str, 
              colonna: str, 
              bins: int = None, 
              seed: int = None, 
              valutation: bool = True, 
              descriptors: list = None,
              plot: bool = True,
              output_dir: str = "figures") -> pd.DataFrame:
    """
    Esegue il campionamento del DataFrame usando il metodo scelto ('random' o 'distribution'),
    valuta la bontà del campionamento confrontando statistiche base con l’originale e
    opzionalmente genera istogrammi + boxplot.

    Parametri:
        df (pd.DataFrame): Dataset originale.
        N (int): Numero di righe da estrarre.
        method (str): 'random' per campionamento casuale, 'distribution' per ponderato.
        colonna (str): Colonna numerica di riferimento.
        bins (int, opzionale): Numero di bin per la discretizzazione (solo per 'distribution').
        seed (int, opzionale): Random seed per riproducibilità.
        valutation (bool): se True stampa media, varianza etc. delle distribuzioni prima e dopo il sampling
        descriptors (list): lista di descrittori della colonna "Description" su cui fare filtro (opzionale)
        plot (bool): se True genera istogrammi + boxplot
        output_dir (str): directory in cui salvare le figure

    Ritorna:
        pd.DataFrame: campione del DataFrame
    """

    np.random.seed(seed)

    # Numero di bin di default
    if bins is None:
        bins = int(np.log2(len(df)) + 1)

    # --- Campionamento ---
    df_clean = df.dropna(subset=[colonna]).copy()
    if method == "random":
        sample = random_sampling(df_clean, N)
    elif method == "distribution":
        sample = distributed_sampling(df_clean, colonna, N, bins)
    else:
        raise ValueError("❌ Metodo non riconosciuto. Usa 'random' o 'distribution'.")

    # --- Valutazione ---
    if valutation:
        print(f"\nℹ Numero righe iniziale: {len(df)}, numero righe campione: {len(sample)}\n")

        def stats(x):
            return {
                'media': x.mean(),
                'mediana': x.median(),
                'stddev': x.std(),
                'p0': np.percentile(x, 0),
                'p25': np.percentile(x, 25),
                'p50': np.percentile(x, 50),
                'p75': np.percentile(x, 75),
                'p100': np.percentile(x, 100)
            }

        x_originale = df[colonna].dropna()
        x_sample = sample[colonna].dropna()

        stats_originale = stats(x_originale)
        stats_sample = stats(x_sample)

        confronto = pd.DataFrame([stats_originale, stats_sample], index=['Originale', 'Campione']).T
        print("📈 Statistiche confronto per la colonna:", colonna)
        print(confronto.round(4))

    # --- Creazione istogrammi ---
    if plot:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.hist(x_originale, bins='sturges', edgecolor='black', alpha=0.7)
        plt.title(f"Istogramma originale: {colonna}")
        plt.xlabel(colonna)
        plt.ylabel("Occorrenze")

        plt.subplot(1,2,2)
        plt.hist(x_sample, bins='sturges', edgecolor='black', alpha=0.7, color='orange')
        plt.title(f"Istogramma campione: {colonna}")
        plt.xlabel(colonna)
        plt.ylabel("Occorrenze")

        plt.tight_layout()
        file_path = os.path.join(output_dir, f"{colonna}_histogram_comparison.png")
        plt.savefig(file_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"📊 Istogrammi salvati in: {file_path}")

    return sample.reset_index(drop=True)
