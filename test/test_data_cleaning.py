from dmp.utils import check_for_column_content, count_word_occurrences
from dmp.data_cleaning import clean_df
import pandas as pd

DATASET_PATH = "data/DM1_game_dataset.csv"

def test_check_for_column_content():
    df = pd.read_csv(DATASET_PATH)

    # Test 1: controlla la colonna 'Name' senza istogramma, tutti i caratteri
    char_list, hist = check_for_column_content(df, 'Name', show_hist=False, only_special_char=False)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 1: {char_list}")

    # Test 1.5: controlla la colonna 'Name' senza istogramma, solo caratteri speciali
    char_list, hist = check_for_column_content(df, 'Name', show_hist=False, only_special_char=True)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 1.5: {char_list}")


    # Test 2: controlla la colonna 'GameWeight' senza istogramma, tutti i caratteri
    char_list, hist = check_for_column_content(df, 'GameWeight', show_hist=False, only_special_char=False)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 2: {char_list}")

    # Test 2.5: controlla la colonna 'GameWeight' senza istogramma, solo caratteri speciali
    char_list, hist = check_for_column_content(df, 'GameWeight', show_hist=False, only_special_char=True)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 2.5: {char_list}")

    # Test 3: controlla la colonna 'YearPublished' senza istogramma, tutti i caratteri
    char_list, hist = check_for_column_content(df, 'YearPublished', show_hist=False, only_special_char=False)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 3: {char_list}")

    # Test 3.5: controlla la colonna 'YearPublished' senza istogramma, solo caratteri speciali
    char_list, hist = check_for_column_content(df, 'YearPublished', show_hist=False, only_special_char=True)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 3.5: {char_list}")

    # Test 4: controlla la colonna 'ComAgeRec' senza istogramma, tutti i caratteri
    char_list, hist = check_for_column_content(df, 'ComAgeRec', show_hist=False, only_special_char=False)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 4: {char_list}")

    # Test 4.5: controlla la colonna 'ComAgeRec' senza istogramma, solo caratteri speciali
    char_list, hist = check_for_column_content(df, 'ComAgeRec', show_hist=False, only_special_char=True)
    assert isinstance(char_list, list)
    assert hist is None
    assert len(char_list) > 0
    print(f"Risultati test 4.5: {char_list}")

def test_count_word_occurrences():
    df = pd.read_csv(DATASET_PATH)
    df = clean_df(df)

    # Test: conta le occorrenze delle parole nella colonna 'Description' senza istogramma
    word_occurrences = count_word_occurrences(df, 'Description', show_hist=False)
    assert isinstance(word_occurrences, dict)
    assert len(word_occurrences) > 0
    print(f"Risultati test count_word_occurrences: {word_occurrences}")
