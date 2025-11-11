#from .clean_description import convert_string_column_to_sets
from dmp.data_cleaning.remove_columns import remove_columns
from .sampling import sample_df
from .transform_columns import min_max_scaling, log_transform
from .merge_ratings_columns import add_weighted_rating
from dmp.data_understanding.analysis_by_descriptors import filter_df_by_descriptors, make_safe_descriptor_name
from dmp.data_understanding import make_hist
import os
import matplotlib.pyplot as plt
from dmp.config import VERBOSE

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


def prepare_df(df, N_samples=None, descriptors=None, hists=False):
    """
    🧹 Opera sul DataFrame pulito e lo prepara per l'analisi,
    tramite le tecniche di 'data preparation'.

    Input: df (DataFrame filtrato)
    Output: df (DataFrame preparato)
    """

    df = df.copy()
    
    # Se specificati, filtra il dataframe in base ai descrittori
    if descriptors:
        df = filter_df_by_descriptors(df, descriptors, column="Description")

    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Inizio processo di Data Preparation...{Colors.RESET}")

    # 🗑️ Rimozione colonne inutili
    df = remove_columns(df, 'ComWeight')
    if VERBOSE:
        print(f"{Colors.GREEN}✅ Rimosse colonna 'ComWeight'.{Colors.RESET}")

    # Make safe name for images
    desc_name = make_safe_descriptor_name(descriptors)
    output_path = f"figures/sampling/{desc_name}"

    # Sampling delle rows
    if descriptors and len(descriptors) > 1:
        df_prepared = sample_df(df, N_samples, method="descriptors", 
        descriptors = descriptors, output_dir=output_path)
    elif N_samples:
        df_prepared = sample_df(df, N_samples, method="random", 
        descriptors = descriptors, output_dir=output_path)
    else:
        df_prepared = df

    # Creazione della colonna Weighted_Ratings (algoritmo IMDB)
    df_prepared = add_weighted_rating(df_prepared, rating_col='Rating',
                                      votes_col='NumUserRatings', new_col='WeightedRating')
    
    # Trasforma colonne in scala logaritmica 
    columns_to_be_tranformed_in_log = ["LanguageEase", "NumOwned", "NumUserRatings", "NumWant", "NumWish"]
    df_prepared = log_transform(df_prepared, columns_to_be_tranformed_in_log)

    # Normalizza la colonna "LanguageEase"
    columns_to_be_normalized = ["LanguageEase"]
    df_prepared = min_max_scaling(df_prepared, columns_to_be_normalized)

    # 🧩 Creazione istogrammi (se richiesto)
    if hists:
        desc_name = make_safe_descriptor_name(descriptors)
        output_path = f"figures/columns_transformed/{desc_name}"

        os.makedirs(output_path, exist_ok=True)
        if VERBOSE:
            print(f"{Colors.YELLOW}📊 Generazione istogrammi in: {output_path}{Colors.RESET}")

        # Seleziona solo le colonne da plottare
        numeric_cols = ["LanguageEase", "NumOwned", "NumUserRatings", "NumWant", "NumWish"]

        for col in numeric_cols:
            titolo = f"Istogramma di {col}_({desc_name}_transformed)"
            plt.close('all')  # Previene overlap di figure
            make_hist(df_prepared, colonna=col, bins='sturges', folder = output_path, titolo=titolo)
        
        #Plotto l'istogramma della colonna "WeightedRating"
        make_hist(df_prepared, colonna="WeightedRating", bins='sturges', folder = output_path, titolo="Istogramma WeightedRating")
        
        if VERBOSE:
            print(f"{Colors.GREEN}✅ Istogrammi creati per {len(numeric_cols)} colonne.{Colors.RESET}")

    # 🏁 Fine
    print(f"\n{Colors.BOLD}{Colors.CYAN}🏁 Preparazione completata con successo!{Colors.RESET}")

    return df_prepared
