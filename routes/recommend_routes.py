from flask import Blueprint, jsonify
from services.recommend_service import get_recommendations

recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    result = get_recommendations(user_id)
    return jsonify(result)
