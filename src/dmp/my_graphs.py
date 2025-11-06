import matplotlib.pyplot as plt
import numpy as np

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

def box_plot(data, percentiles=(5,25,50,75,95), vertical=False, larghezza_boxplot=0.5, y_boxplot=0.5):
    """ Funzione per creare dei 'box and whisker' plot stadardizzati e modulari
    
    Input:
    data: dati da usare per creare il boxplot
    percentiles: tuple con i percentili da usare per i baffi del boxplot
    vertical: se True il boxplot è verticale, altrimenti orizzontale
    larghezza_boxplot: larghezza del boxplot
    y_boxplot: posizione del boxplot sull'asse y (o x se orizzontale)
    
    Output: istanza di oggetto grafico di matplotlib
    """

    plt.boxplot(data, whis=(percentiles[0],percentiles[4]), showfliers=True, 
                vert=vertical, widths=larghezza_boxplot, positions=[y_boxplot], 
                medianprops=dict(color="red", linewidth=1.5))
    return plt


def histo_box(data,colonna):
    """ Funzione per creare un grafico con istogramma e boxplot insieme
    
    Output: istanza di oggetto grafico di matplotlib"""
    # Crea subplot 1x2
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.histo_graph(data=data, binning=30)
    ax1.set_title(f'Istogramma originale di {colonna}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Conteggi")

    ax2=ax1.twinx()
    ax2.set_ylim(0, 1)

  
    ax2.box_plot(data, larghezza_boxplot=0.15, y_boxplot=0.75)
    ax2.get_yaxis().set_visible(False)
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
    ax1.text(0.98, 0.98, pct_text_orig, transform=ax1.transAxes,
             fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    

    return plt


# def plot_column_analysis(df, colonna, bins=30):
#     """
#     Crea un grafico di analisi completo per una colonna numerica con:
#     - Istogramma originale (in alto a sinistra)
#     - Box plot con i percentili 5,25,50,75,95 (in basso a sinistra)
#     - Istogramma pulito senza outliers (in alto a destra)
#     - Box plot pulito senza outliers (in basso a destra)
#     """
#     # Crea subplot 1x2
#     fig = plt.figure(figsize=(15, 10))
#     gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

#     # Prendi dati e rimuovi Nan
#     data = df[colonna].dropna()
#     # Calcola i percentili che si useranno per i box plot
#     percentiles = np.percentile(data, [5, 25, 50, 75, 95])
    
#     # Dati puliti
#     clean_mask = (data >= percentiles[0]) & (data <= percentiles[4])
#     clean_data = data[clean_mask]
#     num_outliers = len(data) - len(clean_data)

#     # Sx: istogramma originale
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.hist(data, bins=bins, edgecolor='black', alpha=0.7)
#     ax1.set_title(f'Istogramma originale di {colonna}')
#     ax1.grid(True, alpha=0.3)
#     ax1.set_ylabel("Conteggi")

#     ax2=ax1.twinx()
#     ax2.set_ylim(0, 1)

#     larghezza_boxplot = 0.15
#     y_boxplot = 0.75    
    
#     ax2.boxplot(data, whis=[5, 95], showfliers=True, 
#                 vert=False, widths=larghezza_boxplot, positions=[y_boxplot], 
#                 medianprops=dict(color="red", linewidth=1.5))

#     ax2.get_yaxis().set_visible(False)

#     # Aggiungi textbox con percentili originali sull'asse del boxplot sinistro
#     pct_text_orig = (
#         f"Percentili (originale):\n"
#         f"5%: {percentiles[0]:.2f}\n"
#         f"25%: {percentiles[1]:.2f}\n"
#         f"50%: {percentiles[2]:.2f}\n"
#         f"75%: {percentiles[3]:.2f}\n"
#         f"95%: {percentiles[4]:.2f}"
#     )
#     ax1.text(0.98, 0.98, pct_text_orig, transform=ax1.transAxes,
#              fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
    

#     # Dx: istogramma pulito
#     ax3 = fig.add_subplot(gs[0, 1])
#     ax3.hist(clean_data, bins=bins, edgecolor='black', alpha=0.7)
#     ax3.set_title(f'Istogramma pulito di {colonna}\n(5-95 percentili)')
#     ax3.grid(True, alpha=0.3)
#     ax3.set_ylabel("Conteggi")

#     ax4 = ax3.twinx()
#     ax4.set_ylim(0, 1)
#     y_boxplot = 0.75 
#     larghezza_boxplot = 0.15


#     ax4.boxplot(clean_data, whis=[5, 95], showfliers=True,
#                 vert=False, widths=larghezza_boxplot, positions=[y_boxplot],
#                 medianprops=dict(color="red", linewidth=1.5))

#     ax4.get_yaxis().set_visible(False)

#     # Aggiungi textbox con percentili puliti e conteggio outliers sull'asse destro
#     if len(clean_data) > 0:
#         percentiles_clean = np.percentile(clean_data, [5, 25, 50, 75, 95])
#         pct_text_clean = (
#             f"Percentili (pulito):\n"
#             f"5%: {percentiles_clean[0]:.2f}\n"
#             f"25%: {percentiles_clean[1]:.2f}\n"
#             f"50%: {percentiles_clean[2]:.2f}\n"
#             f"75%: {percentiles_clean[3]:.2f}\n"
#             f"95%: {percentiles_clean[4]:.2f}\n\n"
#             f"Outliers rimossi: {num_outliers} ({num_outliers/len(data)*100:.2f}%)"
#         )
#     else:
#         pct_text_clean = f"Nessun dato nel range 5-95%. Outliers rimossi: {num_outliers}"

#     ax3.text(0.98, 0.98, pct_text_clean, transform=ax3.transAxes,
#              fontsize=9, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))


#     plt.suptitle(f'Analisi statistica di {colonna}', fontsize=16, y=1.02)

#     # Salva
#     file_path = save_figure(plt, f"analisi_singola_{colonna}", "figures/histograms", ".png")
#     print(f"Analysis plot saved in: {file_path}")
#     plt.close()



