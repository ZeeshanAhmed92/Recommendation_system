from flask import Flask
from .recommend_routes import recommend_user_bp, recommend_genre_bp

def register_blueprints(app: Flask):
    app.register_blueprint(recommend_user_bp)
    app.register_blueprint(recommend_genre_bp)
