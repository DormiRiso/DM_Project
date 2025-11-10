#from .clean_description import convert_string_column_to_sets
from dmp.data_cleaning.remove_columns import remove_columns
from .sampling import sample_df
from .transform_columns import min_max_scaling, log_transform
from .merge_ratings_columns import add_weighted_rating
from dmp.data_understanding.analysis_by_descriptors import filter_df_by_descriptors, make_safe_descriptor_name

# 🎨 Colori ANSI per un output chiaro e leggibile
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def section(title: str, emoji: str = "🧩"):
    """Stampa una sezione evidenziata."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{emoji} {title}{Colors.RESET}")
    print(f"{Colors.HEADER}{'─' * (len(title) + 4)}{Colors.RESET}")


def prepare_df(df, N_samples=None, descriptors=None):
    """
    🧹 Opera sul DataFrame pulito e lo prepara per l'analisi, 
    tramite le tecninche di 'data preparation'.

    Input: df (DataFrame filtrato)
    Output: df (DataFrame preparato)
    """

    df = df.copy()
    
    #Se specificati, filtra il dataframe in base ai descrittori
    if descriptors:
        df = filter_df_by_descriptors(df, descriptors, column="Description")

    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Inizio processo di Data Preparation...{Colors.RESET}")
    print(f"{'=' * 60}\n")

    # Rimuovi colonne dove la correlazione è 1 con altre colonne
    # 🗑️ Rimozione colonne inutili
    section("Rimozione colonne che sono strettamente correlate con altre", "🧺")
    df = remove_columns(df, 'ComWeight')
    print(f"{Colors.GREEN}✅ Rimosse colonna 'ComWeight'.{Colors.RESET}")

    #Make safe name for images
    desc_name = make_safe_descriptor_name(descriptors)
    output_path = f"figures/sampling/{desc_name}"

    # Sampling delle rows
    #df_prepared = sample_df(df, N_samples, "random", "MfgPlaytime",  output_dir= output_path + "_random")
    if N_samples:
        df_prepared = sample_df(df, N_samples, method ="distribution", colonna ="MfgPlaytime", output_dir= output_path+"_distribution")
    else:
        df_prepared = df

    #Creazione della colonna Weighted_Ratings seguendo l'algoritmo di IMDB e Stem
    df_prepared = add_weighted_rating(df_prepared, rating_col='Rating', votes_col='NumUserRatings', new_col='WeightedRating')
    
    #Trasforma colonnein scala logaritmica 
    columns_to_be_tranformed_in_log = ["LanguageEase", "NumOwned", "NumUserRatings", "NumWant", "NumWish"]
    df_prepared = log_transform(df_prepared, columns_to_be_tranformed_in_log)

    #Normalizza la colonna "LanguageEase"
    columns_to_be_normalized = ["LanguageEase"]
    df_prepared = min_max_scaling(df_prepared, columns_to_be_normalized)

 
    # 🏁 Fine
    print(f"\n{Colors.BOLD}{Colors.CYAN}🏁 Preparazione completata con successo!{Colors.RESET}")
    print(f"{'=' * 60}\n")

    return df_prepared
