import numpy as np

import pandas as pd

from sklearn.neighbors import NearestNeighbors

from joblib import dump, load  # import dump functions and load for save and load the model

movies_df = pd.read_csv('../../data/imdb_data/movies.csv')
ratings_df = pd.read_csv('../../data/imdb_data/ratings.csv')

# Unión de datos
movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
# titulos filas, columnas como id de usuario
user_movie_matrix = movie_ratings.pivot_table(index='title', columns='userId', values='rating')

# completar con 0 campos NaN
user_movie_matrix.fillna(0,  inplace=True)

# guardar en archivo csv user_movie_matrix para usarlo en la recomendacion
user_movie_matrix.to_csv('user_movie_matrix.csv')

# crear un modelo KNN con k=5 vecinos cercanos
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
model_knn.fit(user_movie_matrix)

# guardar modelo
dump(model_knn, 'movie_recommendation_trained.joblib')

movie_title = 'Iron Man (2008)' # título de la película de interés
movie_index = np.where(user_movie_matrix.index.values == movie_title)[0][0]
distances, indices = model_knn.kneighbors(user_movie_matrix.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=5)


# imprimir los títulos de las películas recomendadas
print("Películas recomendadas:")
for i in range(1, len(indices.flatten())):
    recommended_movie_title = user_movie_matrix.index.values[indices.flatten()[i]]
    distancia = distances.flatten()[i]
    print(f"{i}: {recommended_movie_title} - {distancia}" )







