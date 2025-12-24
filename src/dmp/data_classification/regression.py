import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsRegressor


def hyperparameter_tuning_single(df_train, df_test, independent_col, dependent_col):
    
    base_dir = f"figures/classification/regression/single"

    # Pulizia Dati
    all_cols = [independent_col] + [dependent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[dependent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[dependent_col].values

    alphas = np.logspace(-3, 4, 20)
    ks = range(1, 99)
    depths = range(1, 31)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Ridge Alpha
    ridge_scores = [r2_score(y_test, Ridge(alpha=a).fit(x_train, y_train).predict(x_test)) for a in alphas]
    axes[0, 0].plot(alphas, ridge_scores, marker='o'); axes[0, 0].set_xscale('log')
    axes[0, 0].set_title("Ridge: R² vs Alpha"); axes[0, 0].set_xlabel("Alpha")

    # 2. Lasso Alpha
    lasso_scores = [r2_score(y_test, Lasso(alpha=a).fit(x_train, y_train).predict(x_test)) for a in alphas]
    axes[0, 1].plot(alphas, lasso_scores, marker='o', color='orange'); axes[0, 1].set_xscale('log')
    axes[0, 1].set_title("Lasso: R² vs Alpha"); axes[0, 1].set_xlabel("Alpha")

    # 3. KNN K
    knn_scores = [r2_score(y_test, KNeighborsRegressor(n_neighbors=k).fit(x_train, y_train).predict(x_test)) for k in ks]
    axes[1, 0].plot(ks, knn_scores, marker='o', color='green')
    axes[1, 0].set_title("KNN: R² vs N Neighbors"); axes[1, 0].set_xlabel("K")

    # 4. Decision Tree Depth
    dt_scores = [r2_score(y_test, DecisionTreeRegressor(max_depth=d).fit(x_train, y_train).predict(x_test)) for d in depths]
    axes[1, 1].plot(depths, dt_scores, marker='o', color='red')
    axes[1, 1].set_title("Tree: R² vs Max Depth"); axes[1, 1].set_xlabel("Depth")

    # Salvataggio
    out_path = os.path.join(base_dir, f"hyperparameter_tuning_{dependent_col}_vs_{independent_col}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.suptitle(f"{dependent_col} vs {independent_col} hyperparameters", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path) 
    plt.close()
    print(f"\nFigura salvata: {out_path}")



def single_regression(
    df_train,
    df_test,
    independent_col,
    dependent_col,
    params = None
):
    """
    Esegue e visualizza i risultati di cinque algoritmi di regressione:
    Banale regressione lineare, e regressione lineare con regolarizzazione Ridge e Lasso, knn e decision trees.

    Il codice pulisce prima i dati rimuovendo le righe con NaN nelle colonne specificate,
    addestra i modelli, stampa le metriche di valutazione (R2, MSE, MAE) e salva i grafici
    dei risultati.

    Args:
        df_train_raw (pd.DataFrame): DataFrame di training originale contenente NaN.
        df_test_raw (pd.DataFrame): DataFrame di test originale contenente NaN.
        independent_col (str): Nome della colonna indipendente (features).
                                E.g., `"NumDesires"`.
        dependent_col (str): Nome della colonna dipendente (target).
                             E.g., `"WeightedRating"`.
        params: Dizionario dei parametri per le funzioni (ridge, lasso, knn, tree)

    """
    if params is None: params = {}

    # Extract params with defaults
    a_ridge = params.get('alpha_ridge', 80)
    a_lasso = params.get('alpha_lasso', 2e-3)
    k_val = params.get('n_neighbors', 20)
    d_val = params.get('max_depth', 6)
    base_dir = f"figures/classification/regression/single"

    # Non considerare le colonne con uno o più NaN
    all_cols = [independent_col] + [dependent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[dependent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[dependent_col].values

    # Definizione dei 5 modelli
    models = [
        ("Linear", LinearRegression(), "Param: None"),
        ("Ridge", Ridge(alpha=a_ridge), f"Alpha: {a_ridge}"),
        ("Lasso", Lasso(alpha=a_lasso), f"Alpha: {a_lasso}"),
        ("KNN", KNeighborsRegressor(n_neighbors=k_val), f"n: {k_val}"),
        ("DecisionTree", DecisionTreeRegressor(max_depth=d_val), f"Max Depth: {d_val}")
    ]
    # Inizializza la figura 5x1
    fig, axes = plt.subplots(5, 1, figsize=(15, 25))
    plt.suptitle(f"Regression Comparison: {dependent_col} vs {independent_col}", fontsize=20)

    for i, (name, model, param_info) in enumerate(models):
        print(f"Training {name}...")
        
        # Addestramento e Predizione
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        # Metriche
        stats_text = (f"{param_info}\n"
                      f"$R^2$: {r2_score(y_test, y_pred):.2f}\n"
                      f"MSE: {mean_squared_error(y_test, y_pred):.2f}\n"
                      f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        

        # Plotting on specific axis
        ax = axes[i]
        sns.scatterplot(ax=ax, data=df_test, x=independent_col, y=dependent_col, 
                            label="True Test", alpha=0.4)
        sns.scatterplot(ax=ax, x=x_test[:,0], y=y_pred, 
                            label="Predicted Test", marker="x", color='red', alpha=0.6)
        
        ax.text(0.05, 0.95, stats_text, 
                transform=ax.transAxes, 
                fontsize=9, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
        
        ax.set_title(name)

    # Titolo principale e salvataggio
    plt.suptitle(f"{dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Spazio per il suptitle
    
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.savefig(out_path) 
    plt.close() 
    print(f"\nFigura salvata: {out_path}")

def hyperparameter_tuning_multiple(df_train, df_test, independent_cols, dependent_col):
    
    base_dir = "figures/classification/regression/multiple"

    if len(independent_cols) != 2:
        print("Errore: questa funzione richiede esattamente 2 colonne indipendenti.")
        return

    # Pulizia Dati
    all_cols = independent_cols + [dependent_col]
    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    X_train = df_train[independent_cols].values
    y_train = df_train[dependent_col].values.ravel()
    X_test = df_test[independent_cols].values
    y_test = df_test[dependent_col].values.ravel()

    alphas = np.logspace(-3, 4, 20)
    ks = range(1, 99)
    depths = range(1, 31)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Ridge Alpha
    ridge_scores = [r2_score(y_test, Ridge(alpha=a).fit(X_train, y_train).predict(X_test)) for a in alphas]
    axes[0, 0].plot(alphas, ridge_scores, marker='o'); axes[0, 0].set_xscale('log')
    axes[0, 0].set_title("Ridge: R² vs Alpha"); axes[0, 0].set_xlabel("Alpha")

    # 2. Lasso Alpha
    lasso_scores = [r2_score(y_test, Lasso(alpha=a).fit(X_train, y_train).predict(X_test)) for a in alphas]
    axes[0, 1].plot(alphas, lasso_scores, marker='o', color='orange'); axes[0, 1].set_xscale('log')
    axes[0, 1].set_title("Lasso: R² vs Alpha"); axes[0, 1].set_xlabel("Alpha")

    # 3. KNN K
    knn_scores = [r2_score(y_test, KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train).predict(X_test)) for k in ks]
    axes[1, 0].plot(ks, knn_scores, marker='o', color='green')
    axes[1, 0].set_title("KNN: R² vs N Neighbors"); axes[1, 0].set_xlabel("K")

    # 4. Decision Tree Depth
    dt_scores = [r2_score(y_test, DecisionTreeRegressor(max_depth=d).fit(X_train, y_train).predict(X_test)) for d in depths]
    axes[1, 1].plot(depths, dt_scores, marker='o', color='red')
    axes[1, 1].set_title("Tree: R² vs Max Depth"); axes[1, 1].set_xlabel("Depth")

    # Salvataggio
    out_path = os.path.join(base_dir, f"hyperparameter_tuning_{dependent_col}_vs_{independent_cols[0]}_{independent_cols[1]}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.suptitle(f"{dependent_col} vs {independent_cols[0]} and {independent_cols[1]} hyperparameters", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path) 
    plt.close()
    print(f"\nFigura salvata: {out_path}")

def multiple_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    independent_cols: list, # Esattamente 2 righe 
    dependent_col: str,
    params: dict = None
):
    """
    Esegue una regressione multipla con due variabili indipendenti e genera
    due subplots per visualizzare l'adattamento del modello rispetto a ciascuna feature.

    Args:
        df_train (pd.DataFrame): DataFrame di training originale.
        df_test (pd.DataFrame): DataFrame di test originale.
        independent_cols (list): Lista di esattamente due nomi di colonne indipendenti (features).
        dependent_col (str): Nome della colonna dipendente (target).
        params: Dizionario dei parametri per le funzioni (ridge, lasso, knn, tree)
    """
    if params is None: params = {}

    # Extract params with defaults
    a_ridge = params.get('alpha_ridge', 1.0)
    a_lasso = params.get('alpha_lasso', 1.0)
    k_val = params.get('n_neighbors', 5)
    d_val = params.get('max_depth', None)

    base_dir = "figures/classification/regression/multiple"

    if len(independent_cols) != 2:
        print("Errore: questa funzione richiede esattamente 2 colonne indipendenti.")
        return

    # Pulizia Dati
    all_cols = independent_cols + [dependent_col]
    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    X_train = df_train[independent_cols].values
    y_train = df_train[dependent_col].values.ravel()
    X_test = df_test[independent_cols].values
    y_test = df_test[dependent_col].values.ravel()

    # Definizione dei 5 modelli
    models = [
        ("Linear", LinearRegression(), "Param: None"),
        ("Ridge", Ridge(alpha=a_ridge), f"Alpha: {a_ridge}"),
        ("Lasso", Lasso(alpha=a_lasso), f"Alpha: {a_lasso}"),
        ("KNN", KNeighborsRegressor(n_neighbors=k_val), f"n: {k_val}"),
        ("DecisionTree", DecisionTreeRegressor(max_depth=d_val), f"Max Depth: {d_val}")
    ]

    # Inizializza la figura 5x2
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    plt.suptitle(f"Multiple Regression Comparison: {dependent_col} vs {independent_cols}", fontsize=20)

    for row_idx, (name, model, param_info) in enumerate(models):
        print(f"Training {name}...")
        
        # Addestramento e Predizione
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metriche
        stats_text = (f"{param_info}\n"
                      f"$R^2$: {r2_score(y_test, y_pred):.2f}\n"
                      f"MSE: {mean_squared_error(y_test, y_pred):.2f}\n"
                      f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

        # Plot per ogni variabile indipendente (Colonna 0 e Colonna 1)
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            
            # Scatter dati reali
            sns.scatterplot(ax=ax, data=df_test, x=independent_cols[col_idx], y=dependent_col, 
                            label="True Test", alpha=0.4)
            
            # Scatter dati predetti
            sns.scatterplot(ax=ax, x=X_test[:, col_idx], y=y_pred, 
                            label="Predicted Test", marker="X", color='red', alpha=0.6)
            
            # Aggiunta box statistiche solo nella prima colonna per chiarezza
            if col_idx == 0:
                ax.text(0.05, 0.95, f"MODEL: {name}\n{stats_text}", 
                        transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
            
            ax.set_title(f"{name}: {dependent_col} vs {independent_cols[col_idx]}")
            ax.legend()

    # Salvataggio
    out_path = os.path.join(base_dir, f"comparison_5x2_{dependent_col}_vs_{independent_cols[0]}_{independent_cols[1]}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.suptitle(f"{dependent_col} vs {independent_cols[0]} and {independent_cols[1]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(out_path) 
    plt.close()
    print(f"\nFigura salvata: {out_path}")

def multivariate_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    independent_cols: list, # Deve contenere esattamente 2 nomi di colonne
    dependent_cols: str
):
    """
    Esegue una regressione multivariata utilizzando Decision Trees e KNN.

    Args:
        df_train (pd.DataFrame): DataFrame di training originale.
        df_test (pd.DataFrame): DataFrame di test originale.
        independent_cols (list): Lista di esattamente due nomi di colonne indipendenti (features).
        dependent_cols (str): Lista di esattamente due nomi di colonne dipendenti (targets).
    """

    # Controllo per garantire che ci siano esattamente due variabili indipendenti per il plot bivariato
    if len(independent_cols) != 2:
        print("Errore: questa funzione richiede esattamente 2 colonne indipendenti per il plotting.")
        return

    if len(dependent_cols) != 2:
        print("Errore: questa funzione richiede esattamente 2 colonne dipendenti per il plotting.")
        return

    base_dir = "figures/classification/regression/multivariate"

    # --- Pulizia Dati ---
    # Rimuove NaN dalle colonne coinvolte nell'addestramento e nel plotting
    all_cols = independent_cols + dependent_cols

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    # --- Preparazione Dati ---
    # X è 2D, Y è 2D (due features in input, due features in output)

    X_train = df_train[independent_cols].values
    y_train = df_train[dependent_cols].values

    X_test = df_test[independent_cols].values
    y_test = df_test[dependent_cols].values

    print("\n\n--- INIZIO: MULTIVARIATE REGRESSION (Decision Tree) ---")
    method = "DecisionTree_Multivariate"
    
    # --- Addestramento ---
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)

    # --- Previsione sui Dati di TEST ---
    y_pred = reg.predict(X_test)
    
    # --- Valutazione (Nota: le metriche R2/MSE sono calcolate sulla media delle due output) ---
    stats_text = (f"Mean $R^2$: {r2_score(y_test, y_pred):.2f}\n"
                f"Mean MSE: {mean_squared_error(y_test, y_pred):.2f}\n"
                f"Mean MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # --- Creazione e Salvataggio dei Subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 6), sharex=True)
    
    fig_title = f"Regressione Multivariata (Test Data): {dependent_cols} vs {independent_cols}"
    plt.suptitle(fig_title, fontsize=16)

    # DataFrame pulito usato per il plotting
    plot_df_test = df_test 

    sns.scatterplot(ax=axes[0, 0], data=plot_df_test, x=independent_cols[0], y=dependent_cols[0], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[0, 0], x=X_test[:, 0].ravel(), y=y_pred[:, 0].ravel(), label="Predicted Test", marker="X", color='red')
    plt.text(0.05, 0.95, stats_text, 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))
    
    axes[0, 0].set_title(f"{dependent_cols[0]} vs {independent_cols[0]}")
    axes[0, 0].legend()

    # Plot [0, 1]: Feature 1 vs Target 2
    sns.scatterplot(ax=axes[0, 1], data=plot_df_test, x=independent_cols[0], y=dependent_cols[1], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[0, 1], x=X_test[:, 0].ravel(), y=y_pred[:, 1].ravel(), label="Predicted Test", marker="X", color='red')
    axes[0, 1].set_title(f"{dependent_cols[1]} vs {independent_cols[0]}")
    axes[0, 1].legend()

    # ITERAZIONE 2: SECONDA VARIABILE INDIPENDENTE 
    # Plot [1, 0]: Feature 2 vs Target 1
    sns.scatterplot(ax=axes[1, 0], data=plot_df_test, x=independent_cols[1], y=dependent_cols[0], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[1, 0], x=X_test[:, 1].ravel(), y=y_pred[:, 0].ravel(), label="Predicted Test", marker="X", color='red')
    axes[1, 0].set_title(f"{dependent_cols[0]} vs {independent_cols[1]}")
    axes[1, 0].legend()

    # Plot [1, 1]: Feature 2 vs Target 2
    sns.scatterplot(ax=axes[1, 1], data=plot_df_test, x=independent_cols[1], y=dependent_cols[1], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[1, 1], x=X_test[:, 1].ravel(), y=y_pred[:, 1].ravel(), label="Predicted Test", marker="X", color='red')
    axes[1, 1].set_title(f"{dependent_cols[1]} vs {independent_cols[1]}")
    axes[1, 1].legend()
    
    # Salvataggio della figura
    out_path = os.path.join(base_dir, f"{method}_2x2_subplots.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path) 
    plt.close()
    print(f"\nFigura salvata: {out_path}")

    print("\n\n--- INIZIO: MULTIVARIATE REGRESSION (KNN) ---")

    method = "KNN_Multivariate"
    reg = KNeighborsRegressor()
    reg.fit(X_train, y_train)

    # --- Previsione sui Dati di TEST ---
    y_pred = reg.predict(X_test)
    
    # --- Valutazione (Nota: le metriche R2/MSE sono calcolate sulla media delle due output) ---
    print('R2: %.2f' % r2_score(y_test, y_pred))
    print('MSE: %.2f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.2f' % mean_absolute_error(y_test, y_pred))

    # --- Creazione e Salvataggio dei Subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 6), sharex=True)
    
    fig_title = f"Regressione Multivariata (Test Data): {dependent_cols} vs {independent_cols}"
    plt.suptitle(fig_title, fontsize=16)

    # DataFrame pulito usato per il plotting
    plot_df_test = df_test 

    sns.scatterplot(ax=axes[0, 0], data=plot_df_test, x=independent_cols[0], y=dependent_cols[0], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[0, 0], x=X_test[:, 0].ravel(), y=y_pred[:, 0].ravel(), label="Predicted Test", marker="X", color='red')
    axes[0, 0].set_title(f"{dependent_cols[0]} vs {independent_cols[0]}")
    axes[0, 0].legend()

    # Plot [0, 1]: Feature 1 vs Target 2
    sns.scatterplot(ax=axes[0, 1], data=plot_df_test, x=independent_cols[0], y=dependent_cols[1], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[0, 1], x=X_test[:, 0].ravel(), y=y_pred[:, 1].ravel(), label="Predicted Test", marker="X", color='red')
    axes[0, 1].set_title(f"{dependent_cols[1]} vs {independent_cols[0]}")
    axes[0, 1].legend()

    # ITERAZIONE 2: SECONDA VARIABILE INDIPENDENTE 
    # Plot [1, 0]: Feature 2 vs Target 1
    sns.scatterplot(ax=axes[1, 0], data=plot_df_test, x=independent_cols[1], y=dependent_cols[0], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[1, 0], x=X_test[:, 1].ravel(), y=y_pred[:, 0].ravel(), label="Predicted Test", marker="X", color='red')
    axes[1, 0].set_title(f"{dependent_cols[0]} vs {independent_cols[1]}")
    axes[1, 0].legend()

    # Plot [1, 1]: Feature 2 vs Target 2
    sns.scatterplot(ax=axes[1, 1], data=plot_df_test, x=independent_cols[1], y=dependent_cols[1], label="True Test", alpha=0.6)
    sns.scatterplot(ax=axes[1, 1], x=X_test[:, 1].ravel(), y=y_pred[:, 1].ravel(), label="Predicted Test", marker="X", color='red')
    axes[1, 1].set_title(f"{dependent_cols[1]} vs {independent_cols[1]}")
    axes[1, 1].legend()
    
    # Salvataggio della figura
    out_path = os.path.join(base_dir, f"{method}_2x2_subplots.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path) 
    plt.close()
    print(f"\nFigura salvata: {out_path}")
