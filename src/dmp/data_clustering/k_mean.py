import numpy as np
import matplotlib.pyplot as plt
from dmp.utils import save_figure
from dmp.config import VERBOSE

class Point:
    """Un punto nello spazio 2D con metodo per assegnare le coordinate randomicamente"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def random_normal(self, mean_x, mean_y, sigma_x, sigma_y):
        self.x = np.random.normal(mean_x, sigma_x)
        self.y = np.random.normal(mean_y, sigma_y)

class Cluster:
    """Un cluster rappresentato dal suo centroide e dalla lista dei punti che contiene."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.points = []

    def add_point(self, point):
        self.points.append(point)

    def mean(self):
        if not self.points:
            return self.x, self.y
        mean_x = np.mean([p.x for p in self.points])
        mean_y = np.mean([p.y for p in self.points])
        return mean_x, mean_y
    
    def update_centroid(self):
        self.x, self.y = self.mean()

    def square_mean_error(self):
        return np.sum([(p.x - self.x)**2 + (p.y - self.y)**2 for p in self.points])


def k_means(x_data, y_data, k, max_steps=5, n_iter=1):
    """Esegue il K-means più volte (n_iter) e restituisce il risultato con SSE più basso."""

    if (len(x_data) != len(y_data)):
        raise ValueError("La lunghezza delle due liste di valori non corrisponde.")

    if not (len(x_data) and len(y_data)):
        raise ValueError(f"Una delle liste {x_data} e {y_data} è vuota.")

    n_points = len(x_data)
    k = min(k, n_points)

    best_sse = np.inf
    best_centroids = None

    for _ in range(n_iter):
        # Inizializza centroidi casuali
        indices = np.random.choice(n_points, k, replace=False)
        centroids = [Cluster(x_data[i], y_data[i]) for i in indices]

        for _ in range(max_steps):
            for c in centroids:
                c.points = []

            for x, y in zip(x_data, y_data):
                distances = np.sqrt((x - np.array([c.x for c in centroids]))**2 +
                                    (y - np.array([c.y for c in centroids]))**2)
                nearest = np.argmin(distances)
                centroids[nearest].add_point(Point(x, y))

            sse = 0
            for c in centroids:
                c.update_centroid()
                sse += c.square_mean_error()

        # Se questa iterazione è migliore, la salvo
        if sse < best_sse:
            best_sse = sse
            best_centroids = [Cluster(c.x, c.y) for c in centroids]
            for i, c in enumerate(centroids):
                best_centroids[i].points = c.points.copy()

    return best_centroids, best_sse


def k_means_scatter(x_column, y_column, k, max_steps=5, n_iter=1, **kwargs):
    """Applica K-means (con n_iter) e produce uno scatter plot con confronto SSE random."""

    x_data, y_data = [], []
    x_column = [float(x) for x in x_column]
    y_column = [float(x) for x in y_column]
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(x)
        y_data.append(y)

    # --- K-means su dati reali ---
    centroids, sse = k_means(x_data, y_data, k, max_steps, n_iter)

    # --- K-means su dati random (stesso range) ---
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    rand_x = np.random.uniform(x_min, x_max, len(x_data))
    rand_y = np.random.uniform(y_min, y_max, len(y_data))
    _, sse_random = k_means(rand_x, rand_y, k, max_steps, n_iter)

    # --- Plot ---
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, c in enumerate(centroids):
        cluster_points = c.points
        ax.scatter([p.x for p in cluster_points],
                   [p.y for p in cluster_points],
                   color=colors[idx % len(colors)],
                   alpha=0.6)

    ax.scatter([c.x for c in centroids],
               [c.y for c in centroids],
               color='black', marker='X', s=120, label='Centroids')

    # --- Legenda con SSE reali e random ---
    ax.scatter([], [], color='none', label=f"SSE reale = {sse:.2f}")
    ax.scatter([], [], color='none', label=f"SSE random = {sse_random:.2f}")
    
    ax.set_title(kwargs.get("title", f"K-means con k={k}"))
    ax.set_xlabel(kwargs.get("x_label", "X"))
    ax.set_ylabel(kwargs.get("y_label", "Y"))
    ax.grid(True)
    ax.legend()

    if VERBOSE:
        print(f"[INFO] Plotted {len(centroids)} clusters (best of {n_iter} runs). "
              f"SSE: {sse:.2f} | Random SSE: {sse_random:.2f}")

    return fig


def sse_vs_k(x_column, y_column, ending_k, max_steps=5, n_iter=1, **kwargs):
    """Calcola SSE per k crescenti, eseguendo più run di K-means per ciascun k."""

    x_data, y_data = [], []
    x_column = [float(x) for x in x_column]
    y_column = [float(x) for x in y_column]
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(x)
        y_data.append(y)

    sse_list = []
    for k in range(1, ending_k):
        _, sse = k_means(x_data, y_data, k, max_steps, n_iter)
        sse_list.append(sse)

    plt.scatter(range(1, ending_k), sse_list, color="black")
    plt.title(kwargs.get("title"))
    plt.xlabel(kwargs.get("x_label"))
    plt.ylabel(kwargs.get("y_label"))

    if VERBOSE:
        print(f"[INFO] Plottati i valori di SSE fino a {ending_k} (best of {n_iter} per k).")

    return plt, sse_list
