from .column_understanding import analizza_colonne_numeriche
from .categories_rankings_stats import number_of_categories_dist, category_couples_heatmap, category_distribution
from .couple_columns_understanding import generate_scatterplots, generate_correlation_heatmap
from dmp.my_graphs import histo_box_grid
from dmp.utils import filter_columns
from .analysis_by_descriptors import filter_df_by_descriptors, make_safe_descriptor_name
import os
import pandas as pd

def understand_df(df_cleaned, df_filtered, do_scatters, do_hists, descriptors= None):
    """
    Funzione che esegue le operazioni di analisi necessarie per il data understanding.
    
    Input:
        df_cleaned: dataframe già pulito
        df_filtered: dataframe già pulito e filtrato dagli outliers
        do_scatters: se True crea gli scatter plot
        do_hists: se True crea gli istogrammi
        descriptors: lista di parole (o singola stringa) per filtrare il df sulla colonna 'Description'
    
    Output:
        ritorna True se non ci sono stati errori
    """

    # Se specificati, filtra il dataframe in base ai descrittori
    if descriptors:
        df_cleaned = filter_df_by_descriptors(df_cleaned, descriptors, column="Description")
        df_filtered = filter_df_by_descriptors(df_filtered, descriptors, column="Description")

    # Analisi della colonna riguardante le categorie dei giochi: 
        # Distribuzione categorie per gioco
    number_of_categories_dist(df_cleaned["Ranks"])
        # Distribuzione occorrenze categorie
    category_distribution(df_cleaned["Ranks"])
        # Heatmap di co-occorrenze di coppie di categorie, normalizzato e non
    category_couples_heatmap(df_cleaned["Ranks"], normalized=False)
    category_couples_heatmap(df_cleaned["Ranks"], normalized=True)

    # Definizione colonne numeriche su cui fare analisi dati

    #### IMPORTANTE: JACOPO HA TOLTO:
    # "MinPlayers", "MaxPlayers",  
    # "NumImplementations", "NumExpansions",  "NumAlternates",
    # 
    ####
    #Colonne su cui fare l'analisi
    columns=[
    "YearPublished", "GameWeight", "ComWeight",  
    "ComAgeRec", "LanguageEase", "NumOwned", "NumWant", "NumWish","MfgPlaytime",
    "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "NumUserRatings",
    ]
        
    # Genera una heatmap per la correlazione di ogni coppia di colonne numeriche
    generate_correlation_heatmap(df_cleaned, columns=columns, output_dir="figures/heatmaps", file_name = "Correlation_Heatmap_unfiltered", title="Matrice di correlazione, dati originali")
    # Definisco il df filtrato dagli outliers, posso chiamarla più volte per filtrare in modo modulare il df e lo salva.
    # Genera una heatmap per la correlazione di ogni coppia di colonne numeriche
    generate_correlation_heatmap(df_filtered, columns=columns, output_dir="figures/heatmaps", file_name = "Correlation_Heatmap_filtered", title="Matrice di correlazione, dati filtrati dagli outliers")
    
    # Rimuovo una delle coppie di colonne correlate 1:1 per poter apprezzare le altre meglio
    uncorrelated_cols = [
        "YearPublished", "GameWeight",   
        "LanguageEase", "NumOwned", "NumWant",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec",
        ]
    generate_correlation_heatmap(df_filtered, columns=uncorrelated_cols, output_dir="figures/heatmaps", file_name = "Correlation_Heatmap_filtered_diminished", title="Matrice di correlazione, dati completamente correlati rimossi")

    # Se richiesto genera una figura composta da tutti gli scatter plot per ogni coppia di colonne numeriche (con e senza outliers)
    if do_scatters:
        desc_name = make_safe_descriptor_name(descriptors)
        output_path = f"figures/scatterplots/{desc_name}"

        # Faccio gli scatterplots del df non filtrato dagli outliers
        generate_scatterplots(df_cleaned, columns, output_dir=output_path, file_name = f"Scatterplots_{desc_name}_unfiltered",
                            title=f"Cleaned Scatterplot Matrix ({desc_name})")


        # Faccio gli scatterplots del df filtrato dagli outliers
        generate_scatterplots(df_filtered, columns, output_dir=output_path, file_name = f"Scatterplots_{desc_name}_filtered",
                            title=f"Filtered Scatterplot Matrix ({desc_name})")

        # In realtà si potrebbe fare anche la scatter matrix di pandas, ma è un po' caotica per tanti dati
        # è mezzanotte non ho voglia di sistemare il filepath se si usa aggiustare:

        #axes = pd.plotting.scatter_matrix(df_cleaned[columns], alpha=0.2, figsize=(15, 15), diagonal='kde')
        #fig = axes[0, 0].get_figure()
        #fig.savefig("scatter_matrix_pandas.png", dpi=300, bbox_inches='tight')

    # Se richiesto genera istogrammi + boxplot per ogni colonna numerica
    if do_hists:
        desc_name = make_safe_descriptor_name(descriptors)
        output_path = f"figures/histograms/{desc_name}"

        # Faccio gli istogrammi delle colonne non filtrate dagli outliers e filtrate dagli outliers, per tutte le colonne numeriche
        histo_box_grid(df_cleaned, columns=None, output_dir=output_path, file_name = f"Histogram_Matrix_{desc_name}_unfiltered",
                    title=f"Unfiltered Histo Boxplot Matrix ({desc_name})",summary=True, extra_info="Outliers non rimossi")

        # Rifaccio il box_grid di istogrammi ma con le colonne filtrate, per tutte le colonne numeriche
        histo_box_grid(df_filtered, columns=None, output_dir=output_path, file_name = f"Histogram_Matrix_{desc_name}_filtered",
                    title=f"Filtered Histo Boxplot Matrix ({desc_name})",summary=True, extra_info="Outliers rimossi")

        analizza_colonne_numeriche(df_cleaned, df_filtered, output_path, columns)

    return True
