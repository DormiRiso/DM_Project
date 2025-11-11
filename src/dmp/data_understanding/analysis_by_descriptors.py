import pandas as pd
import re

import pandas as pd
import re

def filter_df_by_descriptors(df, descriptors, column="Description"):
    """
    Filtra un DataFrame mantenendo solo le righe in cui la colonna specificata
    contiene almeno uno dei descrittori forniti come elemento completo.
    Esempio: se 'Description' contiene "{'war', 'strategy'}" e descriptors = ['war'],
    la riga viene mantenuta.
    """
    if column not in df.columns:
        raise ValueError(f"La colonna '{column}' non esiste nel DataFrame.")
    
    if not descriptors:
        return df  # nessun filtro richiesto
    
    if isinstance(descriptors, str):
        descriptors = [descriptors]
    
    # Converte la colonna in stringhe (es. da set o lista testuale)
    col = df[column].astype(str)

    # Prepara i pattern con gli apici singoli attorno alla parola (es. "'war'")
    # Usiamo regex per evitare corrispondenze parziali come 'warforged'
    patterns = [re.compile(rf"'{re.escape(d.lower())}'") for d in descriptors]

    def match_any(text):
        text_lower = text.lower()
        return any(p.search(text_lower) for p in patterns)

    mask = col.apply(match_any)
    return df[mask]




# Utility per creare un nome sicuro di cartella
def make_safe_descriptor_name(descriptors):
    if descriptors is None:
        return "all_data"
    if isinstance(descriptors, str):
        name = descriptors
    else:
        name = "_".join(descriptors)  # unisci più parole
    # Rimuovi caratteri non alfanumerici o spazi
    name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    return name or "filtered"
