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
    depenent_col
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
        depenent_col (str): Nome della colonna dipendente (target).
                             E.g., `"WeightedRating"`.
    """

    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    # Non considerare le colonne con uno o più NaN
    all_cols = [independent_col] + [depenent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[depenent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[depenent_col].values

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
    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_linear.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=depenent_col)
    plt.plot(x_test, reg.coef_[0]*x_test+reg.intercept_, c="red")

    plt.suptitle(f"Linear Regression, {independent_col} vs {depenent_col}", fontsize=16)
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
    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_linear_ridge.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono


    sns.scatterplot(data=df_test, x=independent_col, y=depenent_col)
    plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")

    plt.suptitle(f"Linear Ridge Regression, {independent_col} vs {depenent_col}", fontsize=16)
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
    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_linear_lasso.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono


    sns.scatterplot(data=df_test, x=independent_col, y=depenent_col)
    plt.plot(x_train, reg.coef_[0]*x_train+reg.intercept_, c="red")

    plt.suptitle(f"Linear Lasso Regression, {independent_col} vs {depenent_col}", fontsize=16)
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
    depenent_col
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
        depenent_col (str): Nome della colonna dipendente (target).
                             E.g., `"WeightedRating"`.
    """

    base_dir = "figures/classification/regression"

    # Non considerare le colonne con uno o più NaN
    all_cols = [independent_col] + [depenent_col]

    df_train = df_train.dropna(subset=all_cols)
    df_test = df_test.dropna(subset=all_cols)

    x_train = df_train[independent_col].values.reshape(-1, 1)
    y_train = df_train[depenent_col].values

    x_test = df_test[independent_col].values.reshape(-1, 1)
    y_test = df_test[depenent_col].values

    print("\n\n--- INIZIO: DECISION TREE REGRESSION ---")
    method = "DecisionTree"
    reg = DecisionTreeRegressor(max_depth = None)
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    base_dir = "figures/classification/regression"
    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_{method}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=depenent_col, label="True")
    sns.scatterplot(data=df_test, x=independent_col, y=reg.predict(x_test).ravel(), label="Predicted", marker="X")
    plt.legend()
    plt.suptitle(f"{method} Regression, {independent_col} vs {depenent_col}", fontsize=16)
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

    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_{method}_Tree.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono
    plt.suptitle(f"{method} Regression, {independent_col} vs {depenent_col} tree", fontsize=16)
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


    out_path = os.path.join(base_dir, f"{depenent_col}_vs_{independent_col}_{method}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Crea le directory se non esistono

    sns.scatterplot(data=df_test, x=independent_col, y=depenent_col, label="True")
    sns.scatterplot(data=df_test, x=independent_col, y=reg.predict(x_test).ravel(), label="Predicted", marker="X")
    plt.legend()
    plt.suptitle(f"{method} Regression, {independent_col} vs {depenent_col}", fontsize=16)
    plt.tight_layout() # Ottimizza lo spazio tra i subplots
    plt.savefig(out_path) 
    plt.close() # Chiude la figura per liberare memoria
    print(f"\nFigura salvata: {out_path}")

def multiple_regression():
    X_train = df_train[["flavanoids", "proline"]].values
    y_train = df_train["od280/od315_of_diluted_wines"].values

    X_test = df_test[["flavanoids", "proline"]].values
    y_test = df_test["od280/od315_of_diluted_wines"].values

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    sns.scatterplot(data=df_test, x="hue", y="od280/od315_of_diluted_wines", label="True")
    sns.scatterplot(data=df_test, x="hue", y=reg.predict(X_test), label="Predicted", marker="X")
    plt.legend()
    plt.show()

def multivariate_regression():

    X_train = df_train[["flavanoids", "proline"]].values
    y_train = df_train[["od280/od315_of_diluted_wines", "ash"]].values

    X_test = df_test[["flavanoids", "proline"]].values
    y_test = df_test[["od280/od315_of_diluted_wines", "ash"]].values

    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print('R2: %.3f' % r2_score(y_test, y_pred))
    print('MSE: %.3f' % mean_squared_error(y_test, y_pred))
    print('MAE: %.3f' % mean_absolute_error(y_test, y_pred))

    y_pred = reg.predict(X_test)

    sns.scatterplot(data=df_test, x="hue", y="od280/od315_of_diluted_wines", label="True")
    sns.scatterplot(data=df_test, x="hue", y=reg.predict(X_test)[:, 0], label="Predicted", marker="X")
    plt.legend()
    plt.show()

    sns.scatterplot(data=df_test, x="hue", y="ash", label="True")
    sns.scatterplot(data=df_test, x="hue", y=reg.predict(X_test)[:, 1], label="Predicted", marker="X")
    plt.legend()
    plt.show()