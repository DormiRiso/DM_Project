from .k_mean import k_means_scatter, sse_vs_k
from .dbscan import dbscan
from .hierarchical import hierarchical_clustering, plot_dendrogram
from dmp.config import VERBOSE
from dmp.utils import save_figure

def cluster_df(df, sse: bool):
    """Funzione che esegue la clusterizzazione dei dati
    """

    # Applico il k-means, sse vs k e dbscan a più combinazioni di colonne del dataset:
    """
    x_columns = ["BestPlayers", "WeightedRating", "AgeRec", "NumDesires", "Weight"]
    y_columns = ["Playtime", "AgeRec", "LanguageEase", "WeightedRating","Playtime"]
    k_list = [5, 5, 5, 5, 5]
    """

    x_columns = ["WeightedRating", "NumDesires"]
    y_columns = [ "AgeRec",  "WeightedRating"]
    k_list = [4, 4]

    for x_column, y_column, k in zip(x_columns, y_columns, k_list):
        # Applico l'aloritmo K-means
        plt1 = k_means_scatter(df[x_column], df[y_column], k, max_steps=10, n_iter = 10, x_label=x_column, y_label=y_column)
        file_path = save_figure(plt1, f'k_means_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        if VERBOSE:
            print(f'Scatter plot del k-means salvato in: {file_path}')
        # Calcolo SSE in funzione di k
        # ATTENZIONE: la scelta di ending_k e max_iters potrebbe portare a runtime elevati
        if sse:
            plt2, _ = sse_vs_k(df[x_column], df[y_column], ending_k=10, max_steps=10, n_iter = 10, x_label="k", y_label="SSE", title=f'SSE vs k per {x_column} vs {y_column}')
            file_path = save_figure(plt2, f'sse_vs_k_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        if VERBOSE:
            print(f'Plot SSE vs k salvato in: {file_path}')
        # Applico l'algoritmo dbscan
        fig, labels = dbscan(x_column=df[x_column], y_column=df[y_column], eps=0.05, min_samples=20, title=f'DBSCAN Clustering di {x_column} vs {y_column}', x_label=x_column, y_label=y_column)
        file_path = save_figure(fig, f'dbscan_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        # Applico l'algoritmo hierarchical
        fig1, labels = hierarchical_clustering(x_column=df[x_column], y_column=df[y_column], n_clusters=4, linkage='average', title="Hierarchical Clustering Example", x_label="Feature X", y_label="Feature Y")
        file_path = save_figure(fig1, f'hierarchical_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
        # Dendrogramma corrispondente
        fig_dendro = plot_dendrogram(x_column=df[x_column], y_column=df[y_column], linkage_method='ward', truncate_mode='lastp', title=f'Dendrogramma - Metodo Ward di {x_column} vs {y_column}')
        file_path = save_figure(fig_dendro, f'Dendrogram_{x_column}_vs_{y_column}', folder="figures/clustered_scatters", extension=".png")
