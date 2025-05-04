import os
import ast
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from flask import Blueprint, jsonify, request
from services.recommend_service import get_recommendations_by_user
from services.recommend_service import get_recommendations_by_genre

recommend_user_bp   = Blueprint("recommend_user", __name__)
recommend_genre_bp  = Blueprint("recommend_genre", __name__)
    
@recommend_user_bp.route('/recommend/user/', methods=['GET'])
def recommend_for_user():
    user_id = request.args.get('user_id')
    result = get_recommendations_by_user(user_id)
    return jsonify(result)

@recommend_genre_bp.route('/recommend/genre', methods=['GET'])
def recommend_for_genre():
    try:
        genre = request.args.get('genre')
        print(f"Genre received: {genre}")  # Debug print

        if not genre:
            return jsonify({'error': 'Missing genre parameter'}), 400

        result = get_recommendations_by_genre(genre)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in recommend_for_genre: {e}")
        return jsonify({'error': 'Internal server error'}), 500
