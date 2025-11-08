from .column_understanding import analizza_colonne_numeriche
from .categories_rankings_stats import number_of_categories_dist, category_couples_heatmap, category_distribution
from .couple_columns_understanding import generate_scatterplots, generate_correlation_heatmap
from dmp.my_graphs import histo_box_grid
from dmp.utils import filter_columns
from .analysis_by_descriptors import filter_df_by_descriptors, make_safe_descriptor_name
import os

def understand_df(df_cleaned, do_scatters, do_hists, descriptors= None):
    """
    Funzione che esegue le operazioni di analisi necessarie per il data understanding.
    
    Input:
        df_cleaned: dataframe già pulito
        do_scatters: se True crea gli scatter plot
        do_hists: se True crea gli istogrammi
        descriptors: lista di parole (o singola stringa) per filtrare il df sulla colonna 'Description'
    
    Output:
        ritorna True se non ci sono stati errori
    """

      #  Se specificati, filtra il dataframe in base ai descrittori
    if descriptors:
        df_cleaned = filter_df_by_descriptors(df_cleaned, descriptors, column="Description")

    # Analisi della colonna riguardante le categorie dei giochi: 
        #distribuzione categorie per gioco
    number_of_categories_dist(df_cleaned["Ranks"])
        #distribuzione occorrenze categorie
    category_distribution(df_cleaned["Ranks"])
        #heatmap di co-occorrenze di coppie di categorie, normalizzato e non
    category_couples_heatmap(df_cleaned["Ranks"], normalized=False)
    category_couples_heatmap(df_cleaned["Ranks"], normalized=True)

    #Definizione colonne numeriche su cui fare analisi dati
    columns=[
        "YearPublished", "GameWeight", "ComWeight", "MinPlayers", "MaxPlayers",
        "ComAgeRec", "LanguageEase", "NumOwned", "NumWant", "NumWish","MfgPlaytime",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "NumUserRatings", "NumAlternates",
        "NumExpansions", "NumImplementations"]
        
    # Genera una heatmap per la correlazione di ogni coppia di colonne numeriche
    generate_correlation_heatmap(df_cleaned, columns=columns, output_dir="figures/heatmaps", file_name = "Correlation_Heatmap_unfiltered")
    #Definisco il df filtrato dagli outliers, posso chiamarla più volte per filtrare in modo modulare il df
    df_filtered = filter_columns(df_cleaned, colonne=columns, method="percentile", params=None, delete_row=False)
    # Genera una heatmap per la correlazione di ogni coppia di colonne numeriche
    generate_correlation_heatmap(df_cleaned, columns=columns, output_dir="figures/heatmaps", file_name = "Correlation_Heatmap_filtered")


    # Se richiesto genera una figura composta da tutti gli scatter plot per ogni coppia di colonne numeriche (con e senza outliers)
    if do_scatters:
        desc_name = make_safe_descriptor_name(descriptors)
        output_path = f"figures/scatterplots/{desc_name}"

        #Faccio gli scatterplots del df non filtrato dagli outliers
        generate_scatterplots(df_cleaned, columns, output_dir=output_path, file_name = f"Scatterplots_{desc_name}_unfiltered",
                            title=f"Cleaned Scatterplot Matrix ({desc_name})")

        #Definisco il df filtrato dagli outliers, posso chiamarla più volte per filtrare in modo modulare il df
        df_filtered = filter_columns(df_cleaned, colonne=columns, method="percentile", params=None, delete_row=False)

        #Faccio gli scatterplots del df filtrato dagli outliers
        generate_scatterplots(df_filtered, columns, output_dir=output_path, file_name = f"Scatterplots_{desc_name}_filtered",
                            title=f"Filtered Scatterplot Matrix ({desc_name})")

    # Se richiesto genera istogrammi + boxplot per ogni colonna numerica
    if do_hists:
        desc_name = make_safe_descriptor_name(descriptors)
        output_path = f"figures/histograms/{desc_name}"

        #Faccio gli istogrammi delle colonne non filtrate dagli outliers e filtrate dagli outliers
        histo_box_grid(df_cleaned, columns=columns, output_dir=output_path, file_name = f"Histogram_Matrix_{desc_name}_unfiltered",
                    title=f"Unfiltered Histo Boxplot Matrix ({desc_name})")
        #Definisco il df filtrato dagli outliers, posso chiamarla più volte per filtrare in modo modulare il df
        df_filtered = filter_columns(df_cleaned, colonne=columns, method="percentile", params=None, delete_row=False)

        #RIfaccio il box_grid di istogrammi ma con le colonne filtrate
        histo_box_grid(df_filtered, columns=columns, output_dir=output_path, file_name = f"Histogram_Matrix_{desc_name}_filtered",
                    title=f"Filtered Histo Boxplot Matrix ({desc_name})")

        analizza_colonne_numeriche(df_cleaned, df_filtered, output_path, columns)

    return True
