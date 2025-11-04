from .clean_description import convert_string_column_to_sets
from .clean_ordered_columns import clean_ordered_columns
from .clean_suggested_players import clean_good_players, clean_best_players
from .convert_wrong_values_into_nan import convert_wrong_values_into_nan
from .remove_columns import remove_columns
from .clean_ranks_and_cats import clean_ranks_and_cats
from .convert_string_column_to_ints import convert_string_column_to_ints

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


def clean_df(df):
    """
    🧹 Pulisce il DataFrame e lo prepara per l'analisi.
    Input: df (DataFrame originale)
    Output: df (DataFrame pulito)
    """

    df = df.copy()

    print(f"\n{Colors.BOLD}{Colors.BLUE}🚀 Inizio processo di Data Cleaning...{Colors.RESET}")
    print(f"{'=' * 60}\n")

    # 🔧 Conversione valori errati in NaN
    section("Pulizia valori errati", "🧪")
    print("👉 Converto 0 e valori non validi in NaN nelle colonne numeriche...")
    columns_to_convert = [
        "YearPublished", "GameWeight", "ComWeight", "MinPlayers", "MaxPlayers",
        "ComAgeRec", "LanguageEase", "BestPlayers", "MfgPlaytime",
        "ComMinPlaytime", "ComMaxPlaytime", "MfgAgeRec", "Rank:strategygames",
        "Rank:abstracts", "Rank:familygames", "Rank:thematic", "Rank:cgs",
        "Rank:wargames", "Rank:partygames", "Rank:childrensgames"
    ]
    df = convert_wrong_values_into_nan(df, columns_to_convert)
    print(f"{Colors.GREEN}✅ Conversione completata!{Colors.RESET}")

    # 📝 Pulizia descrizioni
    section("Pulizia della colonna Description", "🗒️")
    df['Description'] = convert_string_column_to_sets(df, 'Description')
    print(f"{Colors.GREEN}✅ Descrizioni convertite in insiemi di parole!{Colors.RESET}")

    # 👥 Giocatori
    section("Pulizia colonne giocatori", "🎮")
    df = clean_ordered_columns(df, 'MinPlayers', 'MaxPlayers')
    df = clean_good_players(df, 'GoodPlayers', 'MinPlayers', 'MaxPlayers')
    df = clean_best_players(df, 'BestPlayers', 'GoodPlayers')
    print(f"{Colors.GREEN}✅ Colonne giocatori pulite e coerenti!{Colors.RESET}")

    # ⏱️ Tempi di gioco
    section("Pulizia tempi di gioco", "⏱️")
    df = clean_ordered_columns(df, 'ComMinPlaytime', 'ComMaxPlaytime')
    print(f"{Colors.GREEN}✅ Pulizia tempi completata!{Colors.RESET}")

    # 💬 Commenti
    section("Controllo colonna NumComments", "💬")
    if (df['NumComments'] == 0).all():
        df = remove_columns(df, 'NumComments')
        print(f"{Colors.YELLOW}⚠️ Colonna NumComments rimossa (tutti 0).{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✅ NumComments contiene valori utili, mantenuta.{Colors.RESET}")

    # 🗑️ Rimozione colonne inutili
    section("Rimozione colonne inutili", "🧺")
    df = remove_columns(df, 'ImagePath')
    df = remove_columns(df, 'BGGId')
    print(f"{Colors.GREEN}✅ Rimosse colonne 'ImagePath' e 'BGGId'.{Colors.RESET}")

    # 📊 Pulizia colonne rank/cat
    section("Pulizia colonne Rank e Cat", "📊")
    rank_cols = [
        "Rank:strategygames", "Rank:abstracts", "Rank:familygames",
        "Rank:thematic", "Rank:cgs", "Rank:wargames",
        "Rank:partygames", "Rank:childrensgames"
    ]
    cat_cols = [
        "Cat:Strategy", "Cat:Abstract", "Cat:Family",
        "Cat:Thematic", "Cat:CGS", "Cat:War",
        "Cat:Party", "Cat:Childrens"
    ]
    df = clean_ranks_and_cats(df, rank_cols, cat_cols)
    print("📦 Unisco colonne 'cat' e 'rank' in un'unica colonna 'Ranks'.")
    df = remove_columns(df, rank_cols)
    df = remove_columns(df, cat_cols)
    print(f"{Colors.GREEN}✅ Rimosse le colonne originali 'cat' e 'rank'.{Colors.RESET}")

    # 🔢 Conversione a interi
    section("Conversione colonne in Int64", "🔢")
    cols_to_convert_in_int = [
        "YearPublished", "MinPlayers", "MaxPlayers",
        "BestPlayers", "MfgPlaytime", "ComMinPlaytime",
        "ComMaxPlaytime", "MfgAgeRec"
    ]
    df[cols_to_convert_in_int] = df[cols_to_convert_in_int].astype("Int64")
    print(f"{Colors.GREEN}✅ Conversione completata!{Colors.RESET}")

    # 🎯 Conversione Rating
    section("Mappatura Rating in interi", "🎯")
    map_string_to_int = {"Low": -1, "Medium": 0, "High": 1}
    df = convert_string_column_to_ints(df, 'Rating', map_string_to_int)
    print(f"{Colors.GREEN}✅ Colonna 'Rating' convertita in (-1, 0, 1)!{Colors.RESET}")

    # 🏁 Fine
    print(f"\n{Colors.BOLD}{Colors.CYAN}🏁 Pulizia completata con successo!{Colors.RESET}")
    print(f"{'=' * 60}\n")

    return df
