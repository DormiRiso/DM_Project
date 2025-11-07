#from .clean_description import convert_string_column_to_sets
from dmp.data_cleaning.remove_columns import remove_columns
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


def prepare_df(df):
    """
    🧹 Opera sul DataFrame pulito e lo prepara per l'analisi, 
    tramite le tecninche di 'data preparation'.

    Input: df (DataFrame pulito)
    Output: df (DataFrame preparato)
    """

    df = df.copy()

    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Inizio processo di Data Preparation...{Colors.RESET}")
    print(f"{'=' * 60}\n")

    # Rimuovi colonne dove la correlazione è 1 con altre colonne
    # 🗑️ Rimozione colonne inutili
    section("Rimozione colonne che sono strettamente correlate con altre", "🧺")
    df = remove_columns(df, 'ComWeight')
    print(f"{Colors.GREEN}✅ Rimosse colonna 'ComWeight'.{Colors.RESET}")

    # Sampling delle rows ...


    # 🏁 Fine
    print(f"\n{Colors.BOLD}{Colors.CYAN}🏁 Preparazione completata con successo!{Colors.RESET}")
    print(f"{'=' * 60}\n")

    return df
