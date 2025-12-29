from .k_mean import k_means_scatter, sse_vs_k
from .dbscan import dbscan
from .hierarchical import hierarchical_clustering, plot_dendrogram
from dmp.config import VERBOSE
from dmp.utils import save_figure

def cluster_df(df, sse: bool, n_samples: int = None):
    """Funzione che esegue la clusterizzazione dei dati
       senza loop, per permettere tuning manuale dei parametri.
    """

    # ---------------------------------------------------------
    # 0. SAMPLING PRELIMINARE
    # ---------------------------------------------------------
    # Il sampling va fatto UNA volta sola prima di tutto
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        if VERBOSE:
            print(f"Dataset ridotto a {n_samples} righe per l'analisi.")


    # =========================================================
    # CASO 1: WeightedRating vs AgeRec
    # =========================================================
    x_col = "WeightedRating"
    y_col = "AgeRec"
    k_val = 4
    
    # Parametri specifici per questo caso (MODIFICABILI)
    dbscan_eps = 0.03
    dbscan_min_samples = 50
    hierarchical_k = 4
    
    if VERBOSE: print(f"\n--- Elaborazione: {x_col} vs {y_col} ---")
    
    # 1.1 K-means
    plt1 = k_means_scatter(df[x_col], df[y_col], k_val, max_steps=10, n_iter=10, x_label=x_col, y_label=y_col)
    save_figure(plt1, f'k_means_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")

    # 1.2 SSE vs k
    if sse:
        plt2, _ = sse_vs_k(df[x_col], df[y_col], ending_k=10, max_steps=10, n_iter=10, x_label="k", y_label="SSE", title=f'SSE vs k per {x_col} vs {y_col}')
        save_figure(plt2, f'sse_vs_k_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")

    # 1.3 DBSCAN
    fig_db, _ = dbscan(x_column=df[x_col], y_column=df[y_col], eps=dbscan_eps, min_samples=dbscan_min_samples, title=f'DBSCAN {x_col} vs {y_col}', x_label=x_col, y_label=y_col)
    save_figure(fig_db, f'dbscan_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    
    # 1.4 Hierarchical
    fig_hier, _ = hierarchical_clustering(x_column=df[x_col], y_column=df[y_col], n_clusters=hierarchical_k, linkage="average", title=f"Hierarchical {x_col} vs {y_col}", x_label=x_col, y_label=y_col)
    save_figure(fig_hier, f'hierarchical_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    
    # 1.5 Dendrogramma
    fig_dendro = plot_dendrogram(x_column=df[x_col], y_column=df[y_col], linkage_method='average', truncate_mode='lastp', title=f'Dendrogramma {x_col} vs {y_col}')
    save_figure(fig_dendro, f'Dendrogram_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    

    # =========================================================
    # CASO 2: NumDesires vs WeightedRating
    # =========================================================
    x_col = "NumDesires"
    y_col = "WeightedRating"
    k_val = 4

    # Parametri specifici per questo caso
    dbscan_eps = 0.035       
    dbscan_min_samples = 50  
    hierarchical_k = 5

    if VERBOSE: print(f"\n--- Elaborazione: {x_col} vs {y_col} ---")
    
    # 2.1 K-means
    plt1 = k_means_scatter(df[x_col], df[y_col], k_val, max_steps=10, n_iter=10, x_label=x_col, y_label=y_col)
    save_figure(plt1, f'k_means_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")

    # 2.2 SSE vs k
    if sse:
        plt2, _ = sse_vs_k(df[x_col], df[y_col], ending_k=10, max_steps=10, n_iter=10, x_label="k", y_label="SSE", title=f'SSE vs k per {x_col} vs {y_col}')
        save_figure(plt2, f'sse_vs_k_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")

    # 2.3 DBSCAN
    fig_db, _ = dbscan(x_column=df[x_col], y_column=df[y_col], eps=dbscan_eps, min_samples=dbscan_min_samples, title=f'DBSCAN {x_col} vs {y_col}', x_label=x_col, y_label=y_col)
    save_figure(fig_db, f'dbscan_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    
    # 2.4 Hierarchical
    fig_hier, _ = hierarchical_clustering(x_column=df[x_col], y_column=df[y_col], n_clusters=hierarchical_k, linkage="ward", title=f"Hierarchical {x_col} vs {y_col}", x_label=x_col, y_label=y_col)
    save_figure(fig_hier, f'hierarchical_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    
    # 2.5 Dendrogramma
    fig_dendro = plot_dendrogram(x_column=df[x_col], y_column=df[y_col], linkage_method='ward', truncate_mode='lastp', title=f'Dendrogramma {x_col} vs {y_col}')
    save_figure(fig_dendro, f'Dendrogram_{x_col}_vs_{y_col}', folder="figures/clustered_scatters", extension=".png")
    
