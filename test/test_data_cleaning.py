from dmp.utils import check_for_column_content, count_word_occurrences
from dmp.clean_description import convert_string_column_to_sets
from dmp.data_cleaning import clean_df
import pandas as pd
import numpy as np

def test_check_for_column_content():
    """Testa la funzione check_for_column_content."""

    # Dataset di esempio
    df = pd.DataFrame({
        "Description": [
            "Hello, world!",                # Parole con punteggiatura
            "Python 3.10 is great!!!",      # Numeri e punti
            "Data-cleaning @ home",         # Caratteri speciali come "-"
            "Multiple    spaces here",      # Spazi multipli
            "repeat repeat word",           # Parole duplicate
            None,                           # Valore nullo
            1234                            # Valore numerico
        ]
    })

    # Test 1: senza istogramma, tutti i caratteri
    char_list, plot = check_for_column_content(df, "Description", show_hist=False, only_special_char=False)
    assert isinstance(char_list, list)
    assert plot is None
    assert len(char_list) > 0

    # Test 2: senza istogramma, solo caratteri speciali
    char_list, plot = check_for_column_content(df, "Description", show_hist=False, only_special_char=True)
    assert isinstance(char_list, list)
    assert plot is None
    assert all(not c.isalnum() for c in char_list)
    assert len(char_list) > 0

def test_count_word_occurrences():
    """Testa la funzione count_word_occurrences."""

    # Dataset di esempio
    df = pd.DataFrame({
        "Description": [
            "Hello, world!",                # Parole con punteggiatura
            "Python 3.10 is great!!!",      # Numeri e punti
            "Data-cleaning @ home",         # Caratteri speciali come "-"
            "Multiple    spaces here",      # Spazi multipli
            "repeat repeat word",           # Parole duplicate
            None,                           # Valore nullo
            1234                            # Valore numerico
        ]
    })

    df['Description'] = convert_string_column_to_sets(df, 'Description')

    # Test: conta le occorrenze delle parole nella colonna 'Description' senza istogramma
    word_occurrences = count_word_occurrences(df, 'Description', show_hist=False)
    assert isinstance(word_occurrences, dict)
    assert len(word_occurrences) > 0
    print(f"Risultati test count_word_occurrences: {word_occurrences}")

def test_convert_string_column_to_sets():
    """Testa la funzione convert_string_column_to_sets."""

    # Dataset di esempio
    df = pd.DataFrame({
        "Description": [
            "Hello, world!",                # Parole con punteggiatura
            "Python 3.10 is great!!!",      # Numeri e punti
            "Data-cleaning @ home",         # Caratteri speciali come "-"
            "Multiple    spaces here",      # Spazi multipli
            "repeat repeat word",           # Parole duplicate
            None,                           # Valore nullo
            1234                            # Valore numerico
        ]
    })

    # Applica la funzione
    result = convert_string_column_to_sets(df, "Description")

    # Test base: deve essere una Series della stessa lunghezza
    assert isinstance(result, pd.Series)
    assert len(result) == len(df)

    # Tutti gli elementi devono essere insiemi
    assert all(isinstance(x, set) for x in result)

    # Controlli di contenuto logico
    # Caso 1: punteggiatura rimossa
    assert {"Hello", "world"} == result.iloc[0] or {"Hello", "world!"} == result.iloc[0]

    # Caso 2: numeri e punti gestiti correttamente
    assert "Python" in result.iloc[1]
    assert "3" in result.iloc[1] or "3.10" in result.iloc[1] or "3" in " ".join(result.iloc[1])

    # Caso 3: caratteri speciali come '-' e '@' rimossi
    assert "Data" in result.iloc[2]
    assert "cleaning" in " ".join(result.iloc[2]) or "Data-cleaning" in " ".join(result.iloc[2])

    # Caso 4: spazi multipli non influenzano il risultato
    assert "spaces" in result.iloc[3]

    # Caso 5: duplicati rimossi
    assert result.iloc[4] == {"repeat", "word"}

    # Caso 6: valori non stringa → set vuoto
    assert result.iloc[5] == set()
    assert result.iloc[6] == set()

    #  Test finale: tutti gli insiemi contengono solo stringhe
    for s in result:
        assert all(isinstance(el, str) for el in s)

def test_clean_ordered_columns():
    """Testa la funzione clean_ordered_columns."""
    
    from dmp.clean_ordered_columns import clean_ordered_columns
    # Dataset di esempio con vari casi da correggere
    df = pd.DataFrame({
        "lower_column": [0, 2, 5, 3, 4, 1],
        "higher_column": [4, 0, 3, 3, 2, 1]
    })

    # Applica la funzione
    result = clean_ordered_columns(df, 'lower_column', 'higher_column')

    # Deve restituire un DataFrame della stessa forma
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape

    # Test 1: i valori 0 devono diventare np.nan
    assert np.isnan(result.loc[0, "lower_column"])
    assert np.isnan(result.loc[1, "higher_column"])

    #  Test 2: righe con MinPlayers > MaxPlayers (e entrambi > 0) devono essere scambiate
    #   - Riga 2: (5,3) → (3,5)
    #   - Riga 4: (4,2) → (2,4)
    assert result.loc[2, "lower_column"] == 3
    assert result.loc[2, "higher_column"] == 5
    assert result.loc[4, "lower_column"] == 2
    assert result.loc[4, "higher_column"] == 4

    #  Test 3: righe già coerenti restano invariate
    assert result.loc[3, "lower_column"] == 3
    assert result.loc[3, "higher_column"] == 3
    assert result.loc[5, "lower_column"] == 1
    assert result.loc[5, "higher_column"] == 1

    #  Test 4: nessun valore negativo deve restare
    assert (result[["lower_column", "higher_column"]] >= 0).all().all() or np.isnan(result.values).any()

    #  Test 5: tipo di ritorno coerente
    assert list(result.columns) == ["lower_column", "higher_column"]
