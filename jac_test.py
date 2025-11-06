import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def draw_hist(ax, data, binning, **kwargs):
    """Funzione per creare degli istogrammi standardizzati e modulari
    
    Input:
    ax: asse di matplotlib su cui disegnare l'istogramma
    data: dati da usare per creare l'istogramma
    binning: bin da usare per l'istogramma


    Output: n, bins_arr, patches, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    n, bins_arr, patches = ax.hist(data, bins=binning, edgecolor='black', alpha=0.7, **kwargs)
    return n, bins_arr, patches, ax


def box_plot(ax, data, horizontal=True, **kwargs):
    """ Funzione per creare dei 'box and whisker' plot stadardizzati e modulari
    
    Input:
    ax: asse di matplotlib su cui disegnare il boxplot
    data: dati da usare per creare il boxplot
    horizontal: se True crea un boxplot orizzontale
    
    Output: 
    """
    if ax is None:
        fig, ax = plt.subplots()
    bp = ax.boxplot(data, vert= not horizontal, **kwargs)
    return bp, ax


def histo_box(data,colonna,numfigs=1):
    """ Funzione per creare un grafico con istogramma e boxplot insieme
    
    Output: istanza di oggetto grafico di matplotlib"""
    # Crea subplot 1x2
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, numfigs, hspace=0.3, wspace=0.3)
    


    ax_histo = fig.add_subplot(gs[0, 0])
    draw_hist(ax_histo, data=data, binning=30)
    ax_histo.set_title(f'Istogramma originale di {colonna}')
    ax_histo.grid(True, alpha=0.3)
    ax_histo.set_ylabel("Conteggi")

    ax_box=ax_histo.twinx()
    ax_box.set_ylim(0, 1)

    bp = box_plot(ax_box, data, positions=[0.75],widths=0.15,medianprops=dict(color="red", linewidth=1.5))
    print(bp[0]['caps'][0].get_ydata())
    ax_box.get_yaxis().set_visible(False)
    percentiles = np.percentile(data, [5, 25, 50, 75, 95])

    # Aggiungi textbox con percentili originali sull'asse del boxplot sinistro
    pct_text_orig = (
        f"Percentili (originale):\n"
        f"5%: {percentiles[0]:.2f}\n"
        f"25%: {percentiles[1]:.2f}\n"
        f"50%: {percentiles[2]:.2f}\n"
        f"75%: {percentiles[3]:.2f}\n"
        f"95%: {percentiles[4]:.2f}"
    )
    ax_histo.text(0.98, 0.98, pct_text_orig, transform=ax_histo.transAxes,
             fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    return plt


if __name__ == "__main__":
    histo_box(np.random.randn(100), "Esempio_Colonna")