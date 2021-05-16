import numpy as np
import pandas as pd
import ast



def text_to_dict(df):
    dict_columns = ['belongs_to_collection', 'genres', 'spoken_languages', 'production_companies',
                    'production_countries', 'Keywords', 'cast', 'crew']
    for columns in dict_columns:
        df[columns] = df[columns].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    return df

def preprocess(df):
    dict_columns = ['belongs_to_collection', 'genres', 'spoken_languages', 'production_companies',
                    'production_countries', 'Keywords', 'cast', 'crew']

    df = text_to_dict(df)
    df.release_date = pd.to_datetime(df.release_date)
    df['release_year'] = df.release_date.apply(lambda x: x.year)
    df['release_month'] = df.release_date.apply(lambda x: x.month)
    df['director'] = df.crew.apply(lambda x: [y['name'] for y in x if y['job'] == 'Director'])
    df['director'] = df['director'].apply(lambda x: x[0] if len(x) > 0 else None)
    df["budget"] = df["budget"].apply(lambda x: x if x != 0 else None)
    genres = {'Action','Adventure','Animation','Comedy','Crime','Documentary','Drama','Family','Fantasy','History'
        ,'Horror','Music','Mystery','Romance','Science Fiction','TV Movie','Thriller','War','Western'}
    for genre in genres:
        df[genre] = 0
    for index, row in df.iterrows():
        row_genres = row["genres"]
        row_genres = [x["name"] for x in row_genres]
        for genre in genres:
            if genre in row_genres:
                df.at[index, genre] = 1
    df.drop(["genres"], axis='columns', inplace=True)
    cols_to_drop = ['backdrop_path', 'poster_path', 'homepage', 'popularity', 'imdb_id', 'video', 'status'
        , 'original_title', 'overview', 'production_companies', 'production_countries', 'spoken_languages',
                    'title', 'tagline', 'Keywords', 'cast', 'crew']
    df.drop(cols_to_drop, axis='columns', inplace=True)


    return df
