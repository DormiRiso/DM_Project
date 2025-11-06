from .column_understanding import analizza_colonne_numeriche
from .categories_rankings_stats import number_of_categories_dist, category_couples_heatmap, category_distribution
from .couple_columns_understanding import generate_scatterplots, generate_correlation_heatmap

def understand_df(df_cleaned, do_scatters, do_hists):
    """Funzione che esegue le operazioni di analisi necessarie per il data understanding
    
    Input: df_cleaned: dataframe già pulito, do_scatters: se True crea gli scatters, do_hists: se True crea dli istogrammi

    Output: ritorna True se non ci sono stati errori
    """

    # Analisi della colonna riguardante le categorie dei giochi: 
        #distribuzione categorie per gioco
    number_of_categories_dist(df_cleaned["Ranks"])
        #distribuzione occorrenze categorie
    category_distribution(df_cleaned["Ranks"])
        #heatmap di co-occorrenze di coppie di categorie, normalizzato e non
    category_couples_heatmap(df_cleaned["Ranks"], normalized=False)
    category_couples_heatmap(df_cleaned["Ranks"], normalized=True)

    # Genera una heatmap per la correlazione di ogni coppia di colonne numeriche
    columns=[
        "YearPublished", "GameWeight", "ComWeight", "MinPlayers", "MaxPlayers",
        "ComAgeRec", "LanguageEase", "NumOwned", "NumWant", "NumWish","MfgPlaytime",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "NumUserRatings", "NumAlternates",
        "NumExpansions", "NumImplementations"]
    generate_correlation_heatmap(df_cleaned, columns)

    # Se richiesto genera una figura composta da tutti gli scatter plot per ogni coppia di colonne numeriche (con e senza outliers)
    if do_scatters:
        generate_scatterplots(df_cleaned, columns, filter_outliers=None)
        generate_scatterplots(df_cleaned, columns, title="Cleaned Scatterplot Matrix", filter_outliers=(0.05, 0.95))

    # Se richiesto genera istogrammi + boxplot per ogni colonna numerica
    if do_hists:
        analizza_colonne_numeriche(df_cleaned)

    return True
