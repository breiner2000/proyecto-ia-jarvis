from flask import Blueprint, request, jsonify
from app.models import classify_wine_quality


bp = Blueprint('main', __name__)




@bp.route('/')
def index():
    return 'Hola, mundo!'

# use postman -> body -> raw -> json
# wine data from request
# {
#     'volatile acidity': 0.66,
#     'chlorides': 0.029,
#     'free sulfur dioxide': 29,
#     'density': 0.9892,
#     'alcohol': 12.8}
@bp.route('/model/wine', methods=['POST'])
def predict():
    input_data = request.get_json()
    # print(input_data)
    # procesar entrada del usuario y ejecutar modelo
    wine_quality = classify_wine_quality(input_data)

    # Por ahora, respondemos con datos quemados
    output_data = {
        'predicción': 'sí',
        'calidad': str(wine_quality)
    }

    return jsonify(output_data)
