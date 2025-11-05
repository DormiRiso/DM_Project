import matplotlib.pyplot as plt

def bar_graph(x_data, y_data, title: str, x_label: str, y_label:str, size: tuple=(8, 5), log_scale=False):
    """Funzione per creare dei grafici a colonne standardizzati e modulari

    Input: 
    x_data: dati da usare come ticks nell'asse x
    y_data: dati da rappresentare nelle colonne
    x_label: label dell'asse x
    y_label: label dell'asse y
    title: titolo del grafico
    figsize: dimensioni del grafico
    log_scale: se True imposta la scala logaritmica

    Output: instanza di oggetto grafico di matplotlib
    """
    plt.figure(figsize=size)
    plt.bar(x_data, y_data, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(x_label)
    if log_scale:
        plt.yscale("log")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)

    return plt

def hist_graph(data, binning, title: str, x_label: str, y_label:str, size: tuple=(8, 5), log_scale=False):
    """Funzione per creare degli istogrammi standardizzati e modulari
    
    Input:
    data: dati da usare per creare l'istogramma
    binning: bin da usare per l'istogramma
    x_label: label dell'asse x
    y_label: label dell'asse y
    title: titolo del grafico
    figsize: dimensioni del grafico
    log_scale: se True imposta la scala logaritmica

    Output: instanza di oggetto grafico di matplotlib
    """

    plt.figure(figsize=size)
    plt.hist(data, bins=binning, edgecolor='black', align='left')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale("log")
    plt.grid(axis='y', alpha=0.3)

    return plt
