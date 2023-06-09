from flask import Blueprint, request, jsonify
from app.models import classify_wine_quality, predict_car_price, movie_recomendation, predict_fat_percentage,\
    predict_churn, predict_avocado_price, predict_walmart_sales, predict_airlane_delay, predict_hepatitis, predict_covid19, authentication_m
import pandas as pd

import base64
import io
bp = Blueprint('bp', __name__)


@bp.route('/')
def index():
    return 'Hola, mundo!'


# example: use postman -> body -> raw -> json
# wine data from request
# {
#     'volatile acidity': 0.66,
#     'chlorides': 0.029,
#     'free sulfur dioxide': 29,
#     'density': 0.9892,
#     'alcohol': 12.8
#     }

@bp.route('/model/auth', methods=['POST'])
def auth_resp():
    try:
        IMG = request.files['image']
        auth = authentication_m(IMG)
        output_data = {
            'result': 'ok',
            'resp': str(auth)
        }
        return jsonify(output_data)
    except Exception as e:
        # Manejar los errores según sea necesario
        return jsonify({'error': str(e)}), 400


@bp.route('/model/wine', methods=['POST'])
def wine_classifier():
    try:
        input_data = request.get_json()
        # print(input_data)
        # process user input and execute model
        wine_quality = classify_wine_quality(input_data)

        output_data = {
            'result': 'ok',
            'resp': str(wine_quality)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/car', methods=['POST'])
def car_prediction():
    try:
        input_data = request.get_json()
        # print(input_data)
        # process user input and execute model
        car_price = predict_car_price(input_data)

        output_data = {
            'result': 'ok',
            'resp': str(car_price)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/movie', methods=['POST'])
def movie_recommendation():
    try:
        input_data = request.get_json()

        # print(input_data)
        # process user input and execute model
        movie_title = input_data['movie_title']
        movies = movie_recomendation(movie_title)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(movies)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/fat', methods=['POST'])
def fat_prediction():
    try:
        input_data = request.get_json()
        fat_predict = predict_fat_percentage(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(fat_predict)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/churn', methods=['POST'])
def churn_prediction():
    try:
        input_data = request.get_json()
        churn_predict = predict_churn(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(churn_predict)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/avocado', methods=['POST'])
def avocado_prediction():
    try:
        input_data = request.get_json()
        avocado_price_predict = predict_avocado_price(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(avocado_price_predict)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400


@bp.route('/model/sales', methods=['POST'])
def walmart_sales_prediction():
    try:
        input_data = request.get_json()
        walmart_sales_predict = predict_walmart_sales(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(walmart_sales_predict)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400

@bp.route('/model/airlane', methods=['POST'])
def airlines_prediction():
    try:
        input_data = request.get_json()
        airlane_predict_result = predict_airlane_delay(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(airlane_predict_result)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400

@bp.route('/model/hepatitis', methods=['POST'])
def hepatitis_prediction():
    try:
        input_data = request.get_json()
        hepatitis_prediction_result = predict_hepatitis(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(hepatitis_prediction_result)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400

@bp.route('/model/covid19', methods=['POST'])
def covid_prediction():
    try:
        input_data = request.get_json()
        covid19_prediction_result = predict_covid19(input_data)

        # convert list to str
        output_data = {
            'result': 'ok',
            'resp': str(covid19_prediction_result)
        }

        return jsonify(output_data)
    except Exception as e:

        # error
        return jsonify({'error': str(e)}), 400
        # return output_data