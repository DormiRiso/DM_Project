import pandas as pd
import matplotlib.pyplot as plt

from data_cleaning import clean_df

if __name__ == "__main__":
    DATASET_PATH = "data/DM1_game_dataset.csv"
    
    #Carico il df
    df = pd.read_csv(DATASET_PATH)

    #Effettuo il cleaning del dataframe
    df_cleaned = clean_df(df)

    #Crea la versione pulita del dataframe e me lo inserisce nella cartella "data"
    df_cleaned.to_csv("data/cleaned_df.csv", index=False)
