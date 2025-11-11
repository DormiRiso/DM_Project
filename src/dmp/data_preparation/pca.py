import pandas as pd
import numpy as np
from dmp.data_cleaning.remove_columns import remove_columns

def pca(data, columns, newcolumntitle):
    """
    Perform Principal Component Analysis (PCA) on the specified columns of the DataFrame.
    Saves a new column and removes the original ones

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to include in PCA.

    Returns:
        pd.DataFrame: DataFrame with principal components added.
    """

    X = data[columns]
    print("pippo")
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Project data onto the first principal component
    PC1 = X_centered @ eigvecs[:, 0]

    data[f'{newcolumntitle}'] = PC1
    data = remove_columns(data, f'{columns}')

    return data
