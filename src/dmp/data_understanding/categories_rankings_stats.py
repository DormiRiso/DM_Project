from dmp.utils import save_figure
import matplotlib.pyplot as plt

def number_of_categories_dist(ranks_column):

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

    return True

def category_couples_heatmap():
    pass
