TODO: 
- Parser nel main
    - Pulire o no il dataset
    - Data undertanding o no
    - Data clustering o no 
    - Data regression o no

- Data understanding:

    - Colonne "numeriche":
        - Histo, box and whisker plot raw
        - Trovare outliers (tramite i percentili o manualmente)
        - Plotta senza outliers (scrivere numero di punti rimossi nella legenda)

    - Colonne stringe:
        - Da determinare

    - Colonne vettoriali:
        - Histogram del genere singolo
        - Heatmap 8x8

        [1, 1, 1, 0, 0, 0]
        [1, 1, 0, 0, 0, 0]

        diventa: 

        [2, 2, 1, 0, 0, 0
         2, 2, 1, 0, 0, 0
         1, 1, 1, 0, 0, 0
         0, 0, 0, 0, 0 ,0]


        - Calcolo il modulo di ogni vettore per vedere quanto spesso capita che un gioco appartenga a 1,2,3,... tags (histogrammabilissimo)
        - Heatmap delle coppie

    - Case-by-case:
        - Heatmap nxn di min vs max players
            - Pensare come inserire anche info su best e good players vicino
        - Heatmap nxn di min vs max playtime
