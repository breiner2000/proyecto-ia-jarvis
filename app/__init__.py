from flask import Flask


def create_app():
    app = Flask(__name__)

    # Configuración de la base de datos, modelos, etc.

    # Registrar rutas
    from .routes import main
    app.register_blueprint(main.bp)

    return app
