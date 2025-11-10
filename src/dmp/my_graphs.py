import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dmp.utils import save_figure, filter_columns

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
    plt.bar(x_data, y_data, edgecolor='black', alpha=0.7, align='center')
    plt.title(title)
    plt.xlabel(x_label)
    if log_scale:
        plt.yscale("log")
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)

    return plt

def hist_graph(data, title: str, x_label: str, y_label:str, size: tuple=(8, 5), log_scale=False):
    """Funzione per creare degli istogrammi standardizzati e modulari
    default sturges binning
    
    Input:
    data: dati da usare per creare l'istogramma
    x_label: label dell'asse x
    y_label: label dell'asse y
    title: titolo del grafico
    figsize: dimensioni del grafico
    log_scale: se True imposta la scala logaritmica

    Output: instanza di oggetto grafico di matplotlib
    """

    plt.figure(figsize=size)
    plt.hist(data, bins="sturges", edgecolor='black', alpha=0.7, align='mid')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale("log")
    plt.grid(axis='y', alpha=0.3)

    return plt

###############


def draw_hist(ax, data, **kwargs):
    """Funzione per creare degli istogrammi standardizzati e modulari
    
    Input:
    ax: asse di matplotlib su cui disegnare l'istogramma
    data: dati da usare per creare l'istogramma

    default sturges binning

    Output: n, bins_arr, patches, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    n, bins_arr, patches = ax.hist(data, edgecolor='black', alpha=0.7, align='mid', bins='sturges', **kwargs)
    return n, bins_arr, patches, ax


def box_plot(ax, data, horizontal=True, summary=False, extra_info="No dati rimossi", **kwargs):
    """ Funzione per creare dei 'box and whisker' plot stadardizzati e modulari
    
    Input:
    ax: asse di matplotlib su cui disegnare il boxplot
    data: dati da usare per creare il boxplot
    horizontal: se True crea un boxplot orizzontale
    summary: se True aggiunge una textbox con i percentili 5,25,50,75,95
    exrtra_info: Default: 'No dati rimossi'. Se vuoi aggiungere info alla textbox
    
    Output: 
    """
    # sanitize input data: convert to numeric, coerce errors to NaN, drop missing
    s = pd.to_numeric(pd.Series(data), errors='coerce')
    clean = s.dropna().values

    if ax is None:
        fig, ax = plt.subplots()

    # Draw boxplot using cleaned numeric values only
    bp = ax.boxplot(clean if clean.size > 0 else [], whis=[5, 95], vert= not horizontal,
                    medianprops=dict(color="red", linewidth=1.5), **kwargs)

    # Aggiungi textbox con percentili se richiesto (use nan-aware percentile)
    if summary:
        if clean.size == 0:
            pct_text_orig = "Percentili:\nNo numeric data"
        else:
            # use numpy.nanpercentile to be robust to any NaNs (shouldn't be any after dropna)
            percentiles = np.nanpercentile(clean, [5, 25, 50, 75, 95])
            pct_text_orig = (
                f"Percentili:\n"
                f"5%: {percentiles[0]:.2f}\n"
                f"25%: {percentiles[1]:.2f}\n"
                f"50%: {percentiles[2]:.2f}\n"
                f"75%: {percentiles[3]:.2f}\n"
                f"95%: {percentiles[4]:.2f}\n"
                + str(extra_info)
            )

        ax.text(0.98, 0.98, pct_text_orig, transform=ax.transAxes,
                fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))

    return bp, ax

def histo_box(ax, data,colonna, summary=False, extra_info="No dati rimossi"):
    """ Funzione per creare un grafico con istogramma e boxplot insieme
    
    Output: istanza di oggetto grafico di matplotlib"""
    if ax is None:
        fig, ax = plt.subplots()

    draw_hist(ax, data=data)
    ax.set_title(f'Istogramma di {colonna}')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Occorrenze")

    ax_box=ax.twinx()
    ax_box.set_ylim(0, 1)

    box_plot(ax_box, data, positions=[0.75],widths=0.15, summary=summary, extra_info=extra_info)
    ax_box.get_yaxis().set_visible(False)
    return plt

def histo_box_grid(df, columns=None, output_dir="figures/histograms", file_name = "histo_box_matrix", title= "Histo box grid", summary=False, extra_info="No dati rimossi"):
    """" Funzione per creare una griglia di istogrammi + boxplot per ogni colonna numerica del DataFrame
    Input: 
    df: pd.DataFrame contenente i dati
    columns: lista di colonne numeriche da rappresentare. Se None usa tutte le colonne numer
    output_dir: directory in cui salvare la figura
    file_name: nome del file in cui salvare la figura
    title: titolo della figura
    summary: se True aggiunge una textbox con i percentili 5,25,50,75,95 in ogni boxplot
    extra_info: Default: 'No dati rimossi'. Se vuoi aggiungere info alla textbox

    Output: salva la figura in output_dir/file_name.png
    """
    # Se colonna non è specificata, usa tutte le colonne numeriche
    if columns is None:
        df_numeric = df.select_dtypes(include=["number"])
    else:
        df_numeric = df[columns].select_dtypes(include=["number"])
        numfigs= len(columns)

    numeric_cols = df_numeric.columns.tolist()
    numfigs= len(numeric_cols)
    
    # Crea directory di output
    os.makedirs(output_dir, exist_ok=True)

    # Prova a rendere la griglia il più quadrata possibile
    ncols =int(np.floor(np.sqrt(numfigs)))
    nrows =int(np.ceil(np.sqrt(numfigs)))+1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()

    for i in range(numfigs):
        histo_box(axes[i], colonna=numeric_cols[i], data=df_numeric[numeric_cols[i]], summary=summary, extra_info=extra_info)

    # Rimuovi se la griglia non è piena
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    file_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(file_path, dpi=100, bbox_inches="tight")

