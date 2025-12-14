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

def lin_regression(
    df_train,
    df_test,
    independent_col,
    dependent_col
):
    """
    Esegue e visualizza i risultati di tre algoritmi di regressione lineare:
    Banale regressione lineare, e regressione lineare con regolarizzazione Ridge e Lasso.

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
    """

    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    # Non considerare le colonne con uno o più NaN
    all_cols = [independent_col] + [dependent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[dependent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[dependent_col].values

    print("\n\n--- INIZIO: LINEAR REGRESSION ---")

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    
    print('Coefficients: \n', reg.coef_)
    print('Intercept: \n', reg.intercept_)

    y_pred = reg.predict(x_test)

    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))


    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_linear.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=dependent_col)
    plt.plot(x_test, reg.coef_[0]*x_test+reg.intercept_, c="red")

    plt.suptitle(f"Linear Regression, {dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")

    # RIDGE
    method = "Ridge"
    print("\n\n--- INIZIO: LINEAR REGRESSION CON RIDGE---")
    reg = Ridge()
    reg.fit(x_train, y_train)
    print('Coefficients: \n', reg.coef_)
    print('Intercept: \n', reg.intercept_)


    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_linear_ridge.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono


    sns.scatterplot(data=df_test, x=independent_col, y=dependent_col)
    plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")

    plt.suptitle(f"Linear Ridge Regression, {dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")


    y_pred = reg.predict(x_test)

    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    # LASSO
    method = "Lasso"
    print("\n\n--- INIZIO: LINEAR REGRESSION CON LASSO---")
    reg = Lasso()
    reg.fit(x_train, y_train)
    print('Coefficients: \n', reg.coef_)
    print('Intercept: \n', reg.intercept_)


    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_linear_lasso.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono


    sns.scatterplot(data=df_test, x=independent_col, y=dependent_col)
    plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")

    plt.suptitle(f"Linear Lasso Regression, {dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")

    y_pred = reg.predict(x_test)

    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

def nonlin_regression(
    df_train,
    df_test,
    independent_col,
    dependent_col
):
    """
    Esegue e visualizza i risultati di due algoritmi di regressione non-lineare:
    Decision Tree Regressor e K-Nearest Neighbors Regressor (KNN).

    Il codice pulisce prima i dati rimuovendo le righe con NaN nelle colonne specificate,
    addestra i modelli, stampa le metriche di valutazione (R2, MSE, MAE) e salva i grafici
    dei risultati e, per il Decision Tree, una visualizzazione dell'albero.

    Args:
        df_train_raw (pd.DataFrame): DataFrame di training originale contenente NaN.
        df_test_raw (pd.DataFrame): DataFrame di test originale contenente NaN.
        independent_col (str): Nome della colonna indipendente (features).
                                E.g., `"NumDesires"`.
        dependent_col (str): Nome della colonna dipendente (target).
                             E.g., `"WeightedRating"`.
    """

    base_dir = "figures/classification/regression/nonlinear"

    # Non considerare le colonne con uno o più NaN
    all_cols = [independent_col] + [dependent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[dependent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[dependent_col].values

    print("\n\n--- INIZIO: DECISION TREE REGRESSION ---")
    method = "DecisionTree"
    reg = DecisionTreeRegressor(max_depth = None)
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_{method}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=dependent_col, label="True")
    sns.scatterplot(data=df_test, x=independent_col, y=reg.predict(x_test).ravel(), label="Predicted", marker="X", color='red')
    plt.legend()
    plt.suptitle(f"{method} Regression, {dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")

    plt.figure(figsize=(20, 10))

    plot_tree(
        reg,
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2,
        max_depth = 3
    )

    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_{method}_Tree.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono
    plt.suptitle(f"{method} Regression, {dependent_col} vs {independent_col} tree", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")

    # KNN
    method = "KNN"
    print("\n\n--- INIZIO: KNN REGRESSION ---")
    reg = KNeighborsRegressor()
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))


    out_path = os.path.join(base_dir, f"{dependent_col}_vs_{independent_col}_{method}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=dependent_col, label="True")
    sns.scatterplot(data=df_test, x=independent_col, y=reg.predict(x_test).ravel(), label="Predicted", marker="X", color="red")
    plt.legend()
    plt.suptitle(f"{method} Regression, {dependent_col} vs {independent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")


def multiple_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    independent_cols: list, # Deve contenere esattamente 2 nomi di colonne
    dependent_col: str,
    method = "Linear"
):
    """
    Esegue una regressione lineare multipla con due variabili indipendenti e genera
    due subplots per visualizzare l'adattamento del modello rispetto a ciascuna feature.

    Args:
        df_train (pd.DataFrame): DataFrame di training originale.
        df_test (pd.DataFrame): DataFrame di test originale.
        independent_cols (list): Lista di esattamente due nomi di colonne indipendenti (features).
        dependent_col (str): Nome della colonna dipendente (target).
        method (str): Metodo da utilizzare per la regressione (default: 'linear')
    """

    base_dir = "figures/classification/regression/multiple"

    # Controllo per garantire che ci siano esattamente due variabili indipendenti per il plot bivariato
    if len(independent_cols) != 2:
        print("Errore: questa funzione richiede esattamente 2 colonne indipendenti per il plotting.")
        return

    all_cols = independent_cols + [dependent_col]

    # Pulizia Dati: Rimuove le righe con NaN nei set di train e test
    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    X_train = df_train[independent_cols].values
    y_train = df_train[dependent_col].values.ravel()

    X_test = df_test[independent_cols].values
    y_test = df_test[dependent_col].values.ravel()

    print(f"\n\n--- INIZIO: {method} MULTIPLE VARIABLE REGRESSION ---")
    print(f"Target: {dependent_col} vs Features: {independent_cols[0]} e {independent_cols[1]}")

    # Addestramento del modello di regressione lineare

    if method == "Linear":
        reg = LinearRegression()
    elif method == "KNN":
        reg = KNeighborsRegressor()
    elif method == "DecisionTree":
        reg = DecisionTreeRegressor()
    else:
        print("Metodi accettati: 'Linear', 'KNN', 'DecisionTree'")

    reg.fit(X_train, y_train)

    # Previsione sui dati di TEST
    y_pred = reg.predict(X_test)
    
    # Stampa delle metriche di valutazione
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    # --- Creazione dei due subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Titolo generale per l'intera figura
    plt.suptitle(f"Regressione Multipla (Dati Test): {dependent_col} vs {independent_cols[0]} e {independent_cols[1]}", fontsize=16)

    # Plot 1: Feature 1 vs Target
    sns.scatterplot(ax=axes[0], data=df_test, x=independent_cols[0], y=dependent_col, label="True Test", alpha=0.6)
    # Per il plot delle previsioni, usiamo la Feature 1 (indice 0) come X e la y_pred come Y
    sns.scatterplot(ax=axes[0], x=X_test[:, 0], y=y_pred.ravel(), label="Predicted Test", marker="X", color='red')
    axes[0].set_title(f"{dependent_col} vs {independent_cols[0]}")
    axes[0].legend()

    # Plot 2: Feature 2 vs Target
    sns.scatterplot(ax=axes[1], data=df_test, x=independent_cols[1], y=dependent_col, label="True Test", alpha=0.6)
    # Per il plot delle previsioni, usiamo la Feature 2 (indice 1) come X e la y_pred come Y
    sns.scatterplot(ax=axes[1], x=X_test[:, 1], y=y_pred.ravel(), label="Predicted Test", marker="X", color='red')
    axes[1].set_title(f"{dependent_col} vs {independent_cols[1]}")
    axes[1].legend()

    # Salvataggio della figura
    out_path = os.path.join(base_dir, f"{method}_{dependent_col}_vs_{independent_cols[0]}_e_{independent_cols[1]}_subplots.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ottimizza lo spazio tenendo conto del suptitle
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
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

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

    print("\n\n--- INIZIO: MULTIVARIATE REGRESSION (KNN) ---")

    method = "KNN_Multivariate"
    reg = KNeighborsRegressor()
    reg.fit(X_train, y_train)

    # --- Previsione sui Dati di TEST ---
    y_pred = reg.predict(X_test)
    
    # --- Valutazione (Nota: le metriche R2/MSE sono calcolate sulla media delle due output) ---
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

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
