import pandas as pd
import numpy as np
from joblib import load

import os

# get complete route from the file with the trained model
wine_model_path = os.path.join(os.path.dirname(__file__), 'wine_quality', 'wine_quality_trained.joblib')

car_model_path = os.path.join(os.path.dirname(__file__), 'car_price', 'car_price_trained.joblib')

# get complete route from the csv files
user_movie_matrix_path = os.path.join(os.path.dirname(__file__), 'movie_recommendation', 'user_movie_matrix.csv')

movie_model_path = os.path.join(os.path.dirname(__file__), 'movie_recommendation', 'movie_recommendation_trained.joblib')

fat_model_path = os.path.join(os.path.dirname(__file__), 'fat_percentage', 'fat_trained.joblib')

churn_model_path = os.path.join(os.path.dirname(__file__), 'churn_predict', 'churn_trained.joblib')

avocado_model_path = os.path.join(os.path.dirname(__file__), 'avocado_price', 'avocado_price_trained.joblib')

walmart_model_path = os.path.join(os.path.dirname(__file__), 'walmart_sales', 'walmart_sales_trained.joblib')

airlines_path = os.path.join(os.path.dirname(__file__), 'airlines', 'airlines_trained.joblib')

covid19_path = os.path.join(os.path.dirname(__file__), 'covid_19', 'covid19_trained.joblib')

hepatitis_path = os.path.join(os.path.dirname(__file__), 'hepatitis', 'fat_trained.joblib')


def classify_wine_quality(wine_dict):
    try:
        # load the trained model
        rfc = load(wine_model_path)

        # make a df with the new wine
        wine_df = pd.DataFrame([wine_dict])
        wine_df['free sulfur dioxide'] = np.log(1 + wine_df['free sulfur dioxide'])

        # make the prediction
        wine_quality = rfc.predict(wine_df)
        print('La calidad estimada del vino es:', wine_quality)
        # return wine_quality
        return 'La calidad estimada del vino es: ' + str(wine_quality)
    except Exception as e:
        # error
        return f"error: {e}"


def predict_car_price(car_dict):
    try:
        # load the trained model
        rfc = load(car_model_path)
        car_df = pd.DataFrame([car_dict])
        selling_price = rfc.predict(car_df)
        print('El precio de venta estimado es de:', selling_price)
        return 'El precio de venta estimado es de: ' + str(selling_price)
        # return selling_price
    except Exception as e:
        # error
        return f"error: {e}"


def movie_recomendation(movie_title):
    try:
        # load the trained model
        rfc = load(movie_model_path)

        user_movie_matrix = pd.read_csv(user_movie_matrix_path, index_col=0)
        movie_index = np.where(user_movie_matrix.index.values == movie_title)[0][0]
        distances, indices = rfc.kneighbors(user_movie_matrix.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=5)

        # print the movie titles
        recommendations = []
        print("Películas recomendadas:")
        for i in range(1, len(indices.flatten())):
            recommended_movie_title = user_movie_matrix.index.values[indices.flatten()[i]]
            recommendations.append(recommended_movie_title)
            distance = distances.flatten()[i]
            print(f"{i}: {recommended_movie_title} - {distance}")

        str_arr = ', '.join(str(i) for i in recommendations)
        return 'Las películas recomendadas son: ' + str_arr
    except Exception as e:
        # error
        return f"error: {e}"


def predict_fat_percentage(fat_dict):
    try:
        # load the trained model
        rfc = load(fat_model_path)
        new_data = pd.DataFrame(fat_dict)
        data_new = new_data.values

        fat_predict = rfc.predict(data_new)
        print('El % de grasa corporal es:', fat_predict)
        return "El porcentaje de grasa es " + str(int(fat_predict)) + "%"
    except Exception as e:
        # error
        return f"error: {e}"


def predict_churn(churn_dict):
    try:

        churn_model = load(churn_model_path)
        churn_data = pd.DataFrame(churn_dict)
        churn_predict = churn_model.predict(churn_data)

        if churn_predict == 0:
            return "El usuario no es propenso a cambiar de compañia"
        else:
            return "La persona sí es propensa a cambiar de compañía"

    except Exception as e:
        # error
        return f"error: {e}"


def predict_avocado_price(avocado_dict):
    try:
        # load the trained model
        avocado_model = load(avocado_model_path)
        new_data = pd.DataFrame(avocado_dict)

        avocado_price_predict = avocado_model.predict(new_data)
        print('El precio promedio es:', avocado_price_predict)
        return 'El precio promedio es de ' + str(round(avocado_price_predict[0], 2)) + ' dolares'
    except Exception as e:
        # error
        return f"error: {e}"


def predict_walmart_sales(walmart_dict):
    try:
        # load the trained model
        walmart_model = load(walmart_model_path)
        new_data = pd.DataFrame(walmart_dict)

        walmart_sales = walmart_model.predict(new_data)
        print('Las ventas del departamento son: ', walmart_sales)
        return 'Las ventas del departamento son: ' + str(round(walmart_sales[0], 2)) + ' dolares'

    except Exception as e:
        # error
        return f"error: {e}"

def predict_airlane_delay(airlines_dict):
    try:
        # load the trained model
        rfc = load(airlines_path)
        new_data = pd.DataFrame(airlines_dict)
        data_new = new_data.values

        airline_predict = rfc.predict(data_new)
        rounded_num = round(int(airline_predict))
        if rounded_num == -1:
            print('Llegada temprana')
        elif rounded_num == 0:
            print('Llegada exacta')
        elif rounded_num == 1:
            print('Llegada tardía')
    except Exception as e:
        # error
        return f"error: {e}"

def predict_hepatitis(hepatitis_dict):
    try:
        # load the trained model
        rfc = load(hepatitis_path)
        new_data = pd.DataFrame(hepatitis_dict)
        data_new = new_data.values

        hepatitis_predict = rfc.predict(data_new)
        rounded_num = round(int(hepatitis_predict)) % 4

        if rounded_num == 0:
            categoria_str = 'Donante de sangre'
        elif rounded_num == 1:
            categoria_str = 'Hepatitis'
        elif rounded_num == 2:
            categoria_str = 'Fibrosis'
        else:
            categoria_str = 'Cirrosis'

        print(f'La persona pertenece a la categoría: {categoria_str})')
    except Exception as e:
        # error
        return f"error: {e}"

def predict_covid19(covid19_dict):
    try:
        # load the trained model
        rfc = load(covid19_path)
        new_data = pd.DataFrame(covid19_dict)
        data_new = new_data.values

        covid19_predict = rfc.predict(data_new)
        rounded_num = round(int(covid19_predict))

        print(f'Posibles cantidad de defunciones: {rounded_num})')
    except Exception as e:
        # error
        return f"error: {e}"

