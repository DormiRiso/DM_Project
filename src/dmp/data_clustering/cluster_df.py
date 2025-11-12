from .k_mean import k_means_scatter, sse_vs_k
from dmp.config import VERBOSE
from dmp.utils import save_figure
#from sklearn.cluster import DBSCAN

def cluster_df(df):
    """Funzione che esegue la clusterizzazione dei dati
    """

    # Applico il k-means, sse vs k e dbscan a più combinazioni di colonne del dataset:
    x_columns = ["BestPlayers", "WeightedRating", "AgeRec", "NumDesires", "Weight"]
    y_columns = ["Playtime", "AgeRec", "LanguageEase", "WeightedRating","Playtime"]
    k_list = [5, 5, 5, 5, 5]
    for x_column, y_column, k in zip(x_columns, y_columns, k_list):
        plt1 = k_means_scatter(df[x_column], df[y_column], k, max_iters=5, x_label=x_column, y_label=y_column)
        file_path = save_figure(plt1, f'k_means_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        if VERBOSE:
            print(f'Scatter plot del k-means salvato in: {file_path}')
        # ATTENZIONE: la scelta di ending_k e max_iters potrebbe portare a runtime elevati
        """
        plt2, _ = sse_vs_k(df[x_column], df[y_column], ending_k=10, max_iters=10, x_label="k", y_label="SSE", title=f'SSE vs k per {x_column} vs {y_column}')
        file_path = save_figure(plt2, f'sse_vs_k_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        """
        if VERBOSE:
            print(f'Plot SSE vs k salvato in: {file_path}')
