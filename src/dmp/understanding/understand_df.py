import dmp.understanding.make_hist as make_hist

def understand_df(df):
    #faccio l'istogramma della colonna "ratings"
    make_hist.make_hist(df, 'Rating', bins=3, titolo="Istogramma della colonna ratings")