import pandas as pd
import re

def filter_df_by_descriptors(df, descriptors, column="Description"):
    """
    Filtra un DataFrame mantenendo solo le righe in cui la colonna specificata
    contiene una o più parole (descrittori) fornite.
    """
    if column not in df.columns:
        raise ValueError(f"La colonna '{column}' non esiste nel DataFrame.")
    
    if isinstance(descriptors, str):
        descriptors = [descriptors]  # se è una singola parola, la trasformo in lista
    
    # Converte la colonna in stringhe per evitare errori con NaN o tipi misti
    col = df[column].astype(str)

    # Crea una maschera: True se almeno un descrittore è presente nel testo
    mask = col.apply(lambda x: all(d.lower() in x.lower() for d in descriptors))
    
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




