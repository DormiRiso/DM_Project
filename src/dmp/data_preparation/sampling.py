import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dmp.my_graphs import box_plot
from dmp.my_graphs import histo_box_grid

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

def descriptor_weighted_sampling(df: pd.DataFrame, descriptors: list, N: int, seed: int = 42) -> pd.DataFrame:
    """
    Campionamento ponderato basato sui descrittori.
    La probabilità di selezionare una riga è proporzionale
    al numero di descrittori presenti nella colonna 'Description'.

    Stampa anche il numero di elementi campionati per ogni descrittore.
    """
    import numpy as np

    if not descriptors:
        sample = df.sample(n=N, random_state=seed).reset_index(drop=True)
    else:
        if N is None:
            N = len(df)

        # Conta quante volte ciascun descrittore è presente in 'Description'
        def count_matches(desc):
            if isinstance(desc, (list, set)):
                return sum(1 for d in descriptors if d in desc)
            elif isinstance(desc, str):
                return sum(1 for d in descriptors if d.lower() in desc.lower())
            else:
                return 0

        pesi = df['Description'].apply(count_matches).astype(float)

        # Se tutte le righe hanno peso zero, fallback a campionamento casuale
        if pesi.sum() == 0:
            print("⚠️ Nessuna riga contiene i descrittori, sampling casuale eseguito.")
            sample = df.sample(n=N, random_state=seed).reset_index(drop=True)
        else:
            # Normalizza i pesi in probabilità
            pesi /= pesi.sum()

            # Campiona le righe in base ai pesi
            np.random.seed(seed)
            sample_idx = np.random.choice(df.index, size=min(N, len(df)), replace=False, p=pesi.values)
            sample = df.loc[sample_idx].reset_index(drop=True)

    # --- Conta quanti elementi per descrittore nel campione ---
    descr_counts = {}
    for d in descriptors:
        def check_desc(desc):
            if isinstance(desc, (list, set)):
                return d in desc
            elif isinstance(desc, str):
                return d.lower() in desc.lower()
            else:
                return False
        descr_counts[d] = sample['Description'].apply(check_desc).sum()
    
    print("📊 Numero di righe campionate per descrittore:")
    for d, count in descr_counts.items():
        print(f"   - {d}: {count}")

    return sample





def sample_df(df: pd.DataFrame, 
              N: int, 
              method: str, 
              colonne: list = None, 
              bins: int = None, 
              seed: int = 42, 
              valutation: bool = True, 
              descriptors: list = None,
              plot: bool = True,
              output_dir: str = "figures") -> pd.DataFrame:
    """
    Esegue il campionamento del DataFrame usando il metodo scelto ('random', 'distribution' o 'descriptors'),
    valuta la bontà del campionamento e opzionalmente genera istogrammi e boxplot.
    """

    # ✅ Lavora sempre su una copia per evitare effetti collaterali
    df = df.copy()
    np.random.seed(seed)

    # Numero di bin di default
    if bins is None:
        bins = int(np.log2(len(df)) + 1)

    # --- Campionamento ---
    if method == "random":
        sample = random_sampling(df, N, seed)
    elif method == "distribution":
        if not colonne:
            raise ValueError("Per 'distribution' devi fornire almeno una colonna numerica")
        colonna_rif = colonne[0]
        sample = distributed_sampling(df, colonna_rif, N, bins, seed)
    elif method == "descriptors":
        if not descriptors:
            raise ValueError("Per 'descriptors' devi fornire almeno un descrittore")
        sample = descriptor_weighted_sampling(df, descriptors, N, seed)
    else:
        raise ValueError("Metodo non riconosciuto. Usa 'random', 'distribution' o 'descriptors'.")

    # --- Valutazione e grafici ---
    if colonne is not None:
        for colonna in colonne:
            if colonna not in df.columns:
                print(f"⚠️ Colonna '{colonna}' non trovata nel DataFrame originale.")
                continue

            x_originale = df[colonna].dropna()
            x_sample = sample[colonna].dropna()

            if x_originale.empty or x_sample.empty:
                print(f"⚠️ Colonna '{colonna}' vuota dopo la pulizia.")
                continue

            # --- Statistiche ---
            if valutation:
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

                confronto = pd.DataFrame([stats(x_originale), stats(x_sample)], 
                                         index=['Originale', 'Campione']).T
                print(f"\n📈 Statistiche confronto per la colonna: {colonna}")
                print(confronto.round(4))

            # --- Plot ---
            if plot:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{colonna}_histogram_comparison.png")

                # ✅ Usa un nome diverso per non sovrascrivere df!
                df_compare = pd.DataFrame({
                    'originale': x_originale,
                    'sampled': x_sample
                })

                histo_box_grid(
                    df_compare,
                    columns=["originale", "sampled"],
                    output_dir=output_dir,
                    file_name=f"{colonna}_histogram_comparison.png",
                    title=f"Confronto tra sampled e non per: {colonna}",
                    summary=True
                )
                print(f"📊 Istogrammi salvati in: {file_path}")

    print(f"\n✅ Campionamento completato con metodo '{method}' su {len(sample)} righe.")
    return sample.reset_index(drop=True)