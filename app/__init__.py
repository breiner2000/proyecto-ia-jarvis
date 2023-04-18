from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)

    # Configuración de la base de datos, modelos, etc.

    # Registrar rutas
    from .routes import main
    app.register_blueprint(main.bp)

    # Habilitar CORS para todas las rutas
    CORS(app)

    return app
