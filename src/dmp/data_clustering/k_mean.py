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
    """Un cluster rappresentato dal suo centroide e dalla lista dei punti che contiene.
    Contiene metodi per:
        - aggiungere un punto al cluster
        - calcolare il punto medio del cluster
        - aggiornare la posizione del centroide
        - calcolare l'errore quadratico su ogni punto del cluster
    """

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

def k_means(x_data, y_data, k, max_iters=5):
    """Funzione che applica l'algoritmo di K-means clustering
    
    Input:
        x_data: lista di coordinate x dei dati da clusterizzare
        y_data: lista di coordinate y dei dati da clusterizzare
        k: numero di centroidi
        max_iters: numero massimo di terazioni dell'algoritmo, default a 5 per evitare inutili costi computazionali

    Output: restituisce la lista delle posizioni dei centroidi ed il valore dell'errore SSE (somma degli errori quadratici)
    """

    if (len(x_data) != len(y_data)):
        raise ValueError("La lunghezza delle due liste di valori non corriponde, impossibile eseguire il k-means")

    if not (len(x_data) and len(y_data)):
        raise ValueError("Una delle liste di valori è vuota")

    n_points = len(x_data)
    k = min(k, n_points) # Evita di avere più centroidi che punti

    indices = np.random.choice(n_points, k, replace=False)
    centroids = [Cluster(x_data[i], y_data[i]) for i in indices]

    for _ in range(max_iters):
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

    return centroids, sse

def k_means_scatter(x_column, y_column, k, max_iters=5, **kwargs):
    """Funzione che applica l'algoritmo K-means ad un insieme di dati e successivamente produce uno scatter plot per la visualizzazione
    
    Input: 
        x_column: colonna del dataframe corrispondende alla coordinata x
        y_column: colonna del dataframe corrispondende alla coordinata y
        k: numero di centroidi
        max_iters: numero massimo di terazioni dell'algoritmo, default a 5 per evitare inutili costi computazionali
        title: titolo del grafico
        x_label: etichetta dell'asse x
        y_label: etichetta dell'asse y
    
    Output: istanza di oggetto figure di matplotlib.pyplot
    """

    # Gestione degli errori ...

    # Rendo leggibili i dati delle colonne, ignorando una coppia quando uno dei due valori è nan
    x_data = []
    y_data = []
    x_column = [float(x) for x in x_column]
    y_column = [float(x) for x in y_column]
    for x, y in zip(x_column, y_column):
        if np.isnan(x) or np.isnan(y):
            continue
        x_data.append(x)
        y_data.append(y)

    # Eseguo l'algoritmo
    centroids, sse = k_means(x_data, y_data, k, max_iters)

    # Procedo con il plotting
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(figsize=(8, 6))

    # Punti dei cluster
    for idx, c in enumerate(centroids):
        cluster_points = c.points
        ax.scatter([p.x for p in cluster_points],
                   [p.y for p in cluster_points],
                   color=colors[idx % len(colors)],
                   alpha=0.6)

    # Centroidi
    ax.scatter([c.x for c in centroids],
               [c.y for c in centroids],
               color='black', marker='X', s=120, label='Centroids')

    # Aggiungo il valore di SSE come voce della legenda
    ax.scatter([], [], color='none', label=f'SSE = {sse:.2f}')

    ax.set_title(kwargs.get("title"))
    ax.set_xlabel(kwargs.get("x_label"))
    ax.set_ylabel(kwargs.get("y_label"))
    ax.grid(True)
    ax.legend()

    if VERBOSE:
        print(f"[INFO] Plotted {len(centroids)} clusters with their centroids. SSE: {sse:.2f}")

    return fig
