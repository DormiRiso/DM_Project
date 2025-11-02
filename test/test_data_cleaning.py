from dmp.utils import check_for_column_content, count_word_occurrences
from dmp.clean_description import convert_string_column_to_sets
from dmp.clean_ordered_columns import clean_ordered_columns
from dmp.clean_suggested_players import clean_good_players, clean_best_players
from dmp.convert_wrong_values_into_nan import convert_wrong_values_into_nan
from dmp.remove_columns import remove_columns
from dmp.merge_columns_with_prefix import merge_columns_with_prefix
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
    
    # Dataset di esempio con vari casi da correggere
    df = pd.DataFrame({
        "lower_column": [0, 2, 5, 3, 4, 1, "parola"],
        "higher_column": [4, 0, 3, 3, 2, 1, "@d)"]
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

    #  Test 6: parole e caratteri speciali vengono scambiati con nan
    assert pd.isna(result.loc[6, "lower_column"])
    assert pd.isna(result.loc[6, "higher_column"])

    print(result[["lower_column", "higher_column"]])

def test_clean_good_players():
    """Test della funzione clean_good_players."""

    # Dataset di esempio con vari casi da correggere
    df = pd.DataFrame({
        "lower_column": [0, 1, 2, 3, 2, 1, 1, 4],
        "higher_column": [4, 5, 6, 3, 7, 1, 5, 6],
        "good_players": ["g", "4, 5", "7,8", "0", "3, @'; par", "1", "3,4,5,6,7", "1,2,3,4,5"]
    })

    # Applica la funzione
    result = clean_good_players(df, 'good_players', 'lower_column', 'higher_column')

    # Deve restituire un DataFrame della stessa forma
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape

    # Il campo non è un numero -> nan
    assert pd.isna(result.loc[0, "good_players"])
    # Il campo contiene moltiplici valori
    assert result.loc[1, "good_players"] == [4, 5]
    # Il campo è fuori range -> nan
    assert pd.isna(result.loc[2, "good_players"])
    # Il campo è uno 0 -> nan
    assert pd.isna(result.loc[3, "good_players"])
    # Il campo contiene un numero coerente e dei caratteri speciali
    assert result.loc[4, "good_players"] == [3]
    # Il campo è esattamente il numero di giocatori
    assert result.loc[5, "good_players"] == [1]
    # Il campo ha good players maggiori del numero di giocatori
    assert result.loc[6, "good_players"] == [3, 4, 5]
    # Il campo ha good players inferiori al numero di giocatori
    assert result.loc[7, "good_players"] == [4, 5]

    print(result["good_players"])

def test_clean_best_players():
    """Test della funzione clean_best_players."""

    # Dataset di esempio con vari casi da correggere
    df = pd.DataFrame({
        "good_players": ["4,5,6", "4, 5", "7,8", "3"],
        "best_players": ["5", "4, arpofr@@+", "0", "9"]
    })

    # Applica la funzione
    result = clean_best_players(df, 'best_players', 'good_players')

    # Deve restituire un DataFrame della stessa forma
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape

    # Il campo è un numero nell'intervallo corretto
    assert result.loc[0, "best_players"] == [5]
    # Il campo è un numero corretto ma ci sono caratteri non numerici
    assert result.loc[1, "best_players"] == [4]
    # Il campo è uno 0 -> nan
    assert pd.isna(result.loc[2, "best_players"])
    # Il campo è un numero fuori dall'intervallo corretto -> nan
    assert pd.isna(result.loc[3, "best_players"])

    print(result["best_players"])

def test_convert_wrong_values_into_nan():
    # --- Test 1: Valori validi (nessuna sostituzione) ---
    df1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    result1 = convert_wrong_values_into_nan(df1.copy(), 'A')
    expected1 = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    pd.testing.assert_frame_equal(result1, expected1)

    # --- Test 2: Valori uguali a 0 diventano NaN ---
    df2 = pd.DataFrame({'A': [0, 1, 2, 3, 4]})
    result2 = convert_wrong_values_into_nan(df2.copy(), 'A')
    expected2 = pd.DataFrame({'A': [np.nan, 1, 2, 3, 4]})
    pd.testing.assert_frame_equal(result2, expected2)

    # --- Test 3: Valori maggiori del numero di righe diventano NaN ---
    # Qui il DataFrame ha 5 righe, quindi valori > 5 diventano NaN
    df3 = pd.DataFrame({'A': [1, 6, 3, 7, 5]})
    result3 = convert_wrong_values_into_nan(df3.copy(), 'A')
    expected3 = pd.DataFrame({'A': [1, np.nan, 3, np.nan, 5]})
    pd.testing.assert_frame_equal(result3, expected3)

    # --- Test 4: Valori misti (0 e troppo grandi) ---
    df4 = pd.DataFrame({'A': [0, 2, 8, 4, 1]})
    result4 = convert_wrong_values_into_nan(df4.copy(), 'A')
    expected4 = pd.DataFrame({'A': [np.nan, 2, np.nan, 4, 1]})
    pd.testing.assert_frame_equal(result4, expected4)

    # --- Test 5: Più colonne ---
    df5 = pd.DataFrame({
        'A': [0, 2, 6, 4, 1],
        'B': [1, 0, 3, 7, 5]
    })
    result5 = convert_wrong_values_into_nan(df5.copy(), ['A', 'B'])
    expected5 = pd.DataFrame({
        'A': [np.nan, 2, np.nan, 4, 1],
        'B': [1, np.nan, 3, np.nan, 5]
    })
    pd.testing.assert_frame_equal(result5, expected5)

    # --- Test 6: DataFrame vuoto ---
    df6 = pd.DataFrame({'A': []})
    result6 = convert_wrong_values_into_nan(df6.copy(), 'A')
    pd.testing.assert_frame_equal(result6, df6)



def test_remove_columns():
    # --- Test 1: Rimuove una singola colonna ---
    df1 = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })
    result1 = remove_columns(df1.copy(), 'b')
    expected1 = pd.DataFrame({'a': [1, 2, 3], 'c': [7, 8, 9]})
    pd.testing.assert_frame_equal(result1, expected1)

    # --- Test 2: Rimuove più colonne ---
    df2 = pd.DataFrame({
        'a': [1, 2],
        'b': [3, 4],
        'c': [5, 6]
    })
    result2 = remove_columns(df2.copy(), ['a', 'c'])
    expected2 = pd.DataFrame({'b': [3, 4]})
    pd.testing.assert_frame_equal(result2, expected2)

    # --- Test 3: Nessuna colonna rimossa se la lista è vuota ---
    df3 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result3 = remove_columns(df3.copy(), [])
    pd.testing.assert_frame_equal(result3, df3)

    # --- Test 4: Funziona anche se viene passato un singolo nome come stringa ---
    df5 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    result5 = remove_columns(df5.copy(), 'x')
    expected5 = pd.DataFrame({'y': [3, 4]})
    pd.testing.assert_frame_equal(result5, expected5)



def test_merge_columns_with_prefix():
    # --- Test 1: Caso base ---
    df1 = pd.DataFrame({
        'id': [1, 2],
        'cat:sport': [1, 0],
        'cat:music': [0, 1],
        'cat:food':  [1, 1]
    })
    result1 = merge_columns_with_prefix(df1.copy(), prefix="cat:", new_col="categories")
    assert 'categories' in result1.columns
    assert all(isinstance(x, np.ndarray) for x in result1['categories'])
    np.testing.assert_array_equal(result1.loc[0, 'categories'], np.array([1, 0, 1]))
    np.testing.assert_array_equal(result1.loc[1, 'categories'], np.array([0, 1, 1]))
    assert set(result1.columns) == {'id', 'categories'}

    # --- Test 2: Nessuna colonna con il prefisso ---
    df2 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
    result2 = merge_columns_with_prefix(df2.copy(), prefix="cat:", new_col="categories")
    pd.testing.assert_frame_equal(result2, df2)

    # --- Test 3: Colonne tutte 0 ---
    df3 = pd.DataFrame({
        'cat:a': [0, 0, 0],
        'cat:b': [0, 0, 0]
    })
    result3 = merge_columns_with_prefix(df3.copy(), prefix="cat:", new_col="categories")
    np.testing.assert_array_equal(result3.loc[0, 'categories'], np.array([0, 0]))
    np.testing.assert_array_equal(result3.loc[1, 'categories'], np.array([0, 0]))

    # --- Test 4: Prefisso personalizzato ---
    df4 = pd.DataFrame({
        'id': [1, 2],
        'tag:red': [1, 0],
        'tag:blue': [0, 1],
        'tag:green': [1, 1]
    })
    result4 = merge_columns_with_prefix(df4.copy(), prefix='tag:', new_col='tags')
    assert 'tags' in result4.columns
    np.testing.assert_array_equal(result4.loc[0, 'tags'], np.array([1, 0, 1]))
    np.testing.assert_array_equal(result4.loc[1, 'tags'], np.array([0, 1, 1]))

