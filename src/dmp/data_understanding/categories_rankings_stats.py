from dmp.utils import save_figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def number_of_categories_dist(ranks_column):
    """Funzione che crea un istogramma per la distribuzione del numero di categorie rankate per gioco
    
    Input: ranks_cloumn: pandas dataframe column
    
    Output: lengths_list: lista di numero di categorie per ogni record
    """

    lengths_list = []

    for entry in ranks_column:
        #strippo la lista in entrata dagli 0
        rank_list = [x for x in entry if x != 0]
        #aggiungo la sua lunghezza a lenght_lsit
        lengths_list.append(len(rank_list))

    #creo un hist per lenght_list
    plt.figure(figsize=(8, 5))
    plt.hist(lengths_list, bins=range(0, max(lengths_list) + 2), edgecolor='black', align='left')
    plt.title("Distribuzione del numero di categorie per gioco")
    plt.xlabel("Numero di categories")
    plt.ylabel("Occorrenze")
    plt.yscale("log")
    plt.grid(axis='y', alpha=0.3)

    file_path = save_figure(plt, "Distribuzione del numero di categorie per gioco", "figures", ".png")
    print(f'Istogramma per la distribuzione del numero di categorie salvato come: {file_path}')

    return lengths_list

def category_distribution(ranks_column):
    """Funzione che crea un grafico a colonne per il numero di occorrenze di ogni categoria nel dataset

    Input: ranks_cloumn: pandas dataframe column

    Output: category_occ: lista contenente il numero di occorrenze per ogni categoria    
    """

    category_occ = [0]*8

    for entry in ranks_column:
        # Salva gli indici delle categorie non-zero
        active = [i for i, x in enumerate(entry) if x != 0]

        for index in active:
            category_occ[index] += 1

    category_names = ["strategy", "abstract", "family", "thematic", "cgs", "war", "party", "childerns"]

    # Crea grafico a colonne
    plt.figure(figsize=(8, 5))
    plt.bar(category_names, category_occ, color="skyblue", edgecolor="black")
    plt.title("Distribuzione delle categorie")
    plt.xlabel("Categoria")
    plt.ylabel("Occorrenze")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)

    # Salva la figura
    file_path = save_figure(plt, "Distribuzione delle categorie", "figures", ".png")
    print(f"Grafico a colonne della distribuzione delle categorie salvato come: {file_path}")

    return category_occ

def category_couples_heatmap(ranks_column, normalized=False):
    """Funzione che crea una heatmap per le coppie di categorie co-occorrenti.

    Input:
        ranks_column: colonna di un DataFrame contenente liste o array di categorie (0 = non attiva)
        normalized: se True, normalizza la matrice di co-occorrenza

    Output:
        matrix: matrice NxN delle occorrenze o co-occorrenze normalizzate
    """

    n_categories = 8
    matrix = np.zeros((n_categories, n_categories), dtype=float)

    # Conta le co-occorrenze
    for entry in ranks_column:
        active = [i for i, x in enumerate(entry) if x != 0]
        for i in active:
            for j in active:
                matrix[i, j] += 1

    # Normalizzazione opzionale (cosine-like)
    if normalized:
        for i in range(n_categories):
            for j in range(n_categories):
                if matrix[i, i] > 0 and matrix[j, j] > 0:
                    matrix[i, j] = matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])

    category_names = ["strategy", "abstract", "family", "thematic", "cgs", "war", "party", "childerns"]

    plt.figure(figsize=(7, 6))

    # Heatmap: logaritmica solo se non normalizzata
    if normalized:
        im = plt.imshow(matrix, cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(im, label="Co-occorrenza normalizzata")
        plt.title("Heatmap della co-occorrenza delle categorie (normalizzata)")
    else:
        vmin = max(1, matrix.min())  # evita log(0)
        vmax = max(1, matrix.max())
        im = plt.imshow(matrix, cmap="cividis", norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")
        plt.colorbar(im, label="Occorrenze (scala log)")
        plt.title("Heatmap logaritmica della co-occorrenza delle categorie")

    # Etichette
    plt.xticks(ticks=np.arange(n_categories), labels=category_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(n_categories), labels=category_names)
    plt.xlabel("Categoria")
    plt.ylabel("Categoria")

    # Annotazioni celle
    for i in range(n_categories):
        for j in range(n_categories):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            plt.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    # Salvataggio figura
    filename = "category_couples_heatmap_norm" if normalized else "category_couples_heatmap"
    file_path = save_figure(plt, filename, "figures", ".png")
    print(f"Heatmap per le coppie di categorie salvata in: {file_path}")

    return matrix
