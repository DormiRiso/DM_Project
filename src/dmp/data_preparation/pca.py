import pandas as pd
import numpy as np
from dmp.data_cleaning.remove_columns import remove_columns
from sklearn.decomposition import PCA

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
    #pca non accetta i nan quindi rimpiazza con la media della colonna
    X = data[columns].fillna(data[columns].mean())
    pca = PCA(n_components=1)  # keep only the first principal component
    X_pca = pca.fit_transform(X)


    data[newcolumntitle] = X_pca
    # Pass the columns list directly to remove_columns (don't stringify it)
    data = remove_columns(data, columns)

    return data