def remove_columns(df, cols):
    """
    Rimuove una o più colonne da un DataFrame pandas.

    Parametri
    ----------
    df : pandas.DataFrame
        Il DataFrame da cui rimuovere le colonne.
    cols : str o list
        Nome (o lista di nomi) delle colonne da eliminare.

    Ritorna
    -------
    pandas.DataFrame
        Una nuova copia del DataFrame senza le colonne specificate.
    """
    
    # Rimuove le colonne indicate dall'argomento `cols`.
    df = df.drop(cols, axis=1)

    # Restituisce il DataFrame aggiornato senza le colonne rimosse.
    return df

