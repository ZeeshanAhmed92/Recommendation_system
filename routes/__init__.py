from flask import Flask
from .recommend_routes import recommend_bp

def register_blueprints(app: Flask):
    app.register_blueprint(recommend_bp)