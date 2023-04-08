from flask import Blueprint, request, jsonify
from app.models import classify_wine_quality, predict_car_price, movie_recomendation, predict_fat_percentage
import pandas as pd

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
@bp.route('/model/wine', methods=['POST'])
def wine_classifier():
    try:
        input_data = request.get_json()
        # print(input_data)
        # process user input and execute model
        wine_quality = classify_wine_quality(input_data)

        output_data = {
            'result': 'ok',
            'quality': str(wine_quality)
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
            'price': str(car_price)
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
            'movies': str(movies)
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
            'fat_%': str(fat_predict)
        }

        return jsonify(output_data)
    except Exception as e:
        # error
        return jsonify({'error': str(e)}), 400
