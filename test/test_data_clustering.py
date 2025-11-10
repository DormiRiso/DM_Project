import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dmp.data_clustering import k_means_scatter

def test_k_means_scatter():

    # === Simulo un dataset come se fosse stato letto da un CSV ===
    #np.random.seed(42)
    
    data = {
        "Feature_X": np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(5, 1, 20),
            np.random.normal(-5, 1, 20)
        ]),
        "Feature_Y": np.concatenate([
            np.random.normal(0, 1, 20),
            np.random.normal(5, 1, 20),
            np.random.normal(-5, 1, 20)
        ])
    }

    # Creo un DataFrame come se provenisse da pandas.read_csv("dataset.csv")
    df = pd.DataFrame(data)

    # Inserisco qualche valore NaN per testare la gestione dei dati mancanti
    df.loc[5, "Feature_X"] = np.nan
    df.loc[10, "Feature_Y"] = np.nan

    # === Chiamo la funzione da testare ===
    fig = k_means_scatter(
        df["Feature_X"],
        df["Feature_Y"],
        k=3,
        max_iters=10,
        title="Test Cluster Plot (CSV Data)",
        x_label="Feature X",
        y_label="Feature Y"
    )

    # === Verifiche automatiche ===

    # Deve restituire una figura matplotlib valida
    assert isinstance(fig, Figure), "k_means_scatter() should return a matplotlib Figure"

    # La legenda deve esistere e contenere l’SSE
    legend = fig.axes[0].get_legend()
    assert legend is not None, "Legend should exist"
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("SSE" in lbl for lbl in labels), "Legend should contain SSE value"

    # Titolo e label corretti
    ax = fig.axes[0]
    assert ax.get_title() == "Test Cluster Plot (CSV Data)"
    assert ax.get_xlabel() == "Feature X"
    assert ax.get_ylabel() == "Feature Y"

    # (Opzionale) salvo la figura per debug se fallisce
    fig.savefig("test/test_figures/test_cluster_plot_from_csv.png")

    plt.close(fig)
