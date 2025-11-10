from .k_mean import k_means_scatter
from dmp.config import VERBOSE
from dmp.utils import save_figure

def cluster_df(df):
    """Funzione che esegue la clusterizzazione dei dati
    """

    # Applico il k-means e plot a più combinazioni di colonne del dataset:
    x_columns = ["GameWeight", "LanguageEase"]
    y_columns = ["ComWeight", "NumOwned"]
    k_list = [5, 5]
    for x_column, y_column, k in zip(x_columns, y_columns, k_list):
        plt = k_means_scatter(df[x_column], df[y_column], k, max_iters=5, x_label=x_column, y_label=y_column)
        file_path = save_figure(plt, f'k_means_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        if VERBOSE:
            print(f'Scatter plot del k-means salvato in: {file_path}')
