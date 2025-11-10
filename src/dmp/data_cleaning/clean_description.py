import re
from dmp.utils import check_for_column_content

def convert_string_column_to_sets(df, column_name):
    """
    Converte una colonna di stringhe di un DataFrame in insiemi di parole uniche.
    Utilizza check_for_column_content per determinare i caratteri da rimuovere.
    """

    # Determina i caratteri speciali da eliminare
    chars_to_remove = check_for_column_content(df, column_name, only_special_char=True)[0]
    # Crea una regex (espressione regolare) che identifica tutti i caratteri da rimuovere.
    # Le regex sono schemi usati per cercare o sostituire testo; qui "[" + ... + "]" indica "uno qualsiasi di questi caratteri".
    # re.escape serve a proteggere i caratteri speciali (come [, ], {, }) così vengono trattati come testo normale.
    pattern = "[" + re.escape(''.join(chars_to_remove)) + "]"

    # Applica la conversione riga per riga
    def clean_and_split(text):
        if not isinstance(text, str):
            return set()
        cleaned = re.sub(pattern, " ", text)
        words = [w for w in cleaned.strip().split() if w]
        return set(words)

    # Ritorna una Serie con gli insiemi
    return df[column_name].apply(clean_and_split)

