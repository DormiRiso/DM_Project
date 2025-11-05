from dmp.utils import save_figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def category_couples_heatmap(ranks_column):
    """Funzione che crea una heatmap per le coppie di categorie co-occorrenti

    Input: ranks_cloumn: pandas dataframe column

    Output: matrix: matrice contenente le occorrenze di ogni coppia di categorie (lungo la diagonale il numero di occorrenze per singola categoria) 
    """

    n_categories = 8
    matrix = np.zeros((n_categories, n_categories), dtype=int)

    for entry in ranks_column:
        # Salva gli indici delle categorie non-zero
        active = [i for i, x in enumerate(entry) if x != 0]

        # Per ogni coppia di indici incremente il contatore nella matrice
        for i in active:
            for j in active:
                matrix[i, j] += 1

    category_names = ["strategy", "abstract", "family", "thematic", "cgs", "war", "party", "childerns"]

    # plot heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=category_names,
        yticklabels=category_names
    )
    plt.title("Heatmap della co-occorrenza delle categorie")
    plt.xlabel("Categoria")
    plt.ylabel("Categoria")

    file_path = save_figure(plt, "category_couples_heatmap", "figures", ".png")
    print(f'Heatmap per le coppie di categorie salvata in: {file_path}')

    return matrix
