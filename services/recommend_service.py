import os
import ast
import logging
import warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR

# Optionally disable oneDNN verbose logs (as mentioned in the logs)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Only if you want to turn off oneDNN

# Suppress Flask/Werkzeug logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Suppress Python user warnings
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

# Load model once
ncf_model = load_model('ncf_model.h5')

# DB credentials
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# SQLAlchemy engine
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:3306/{db_name}"
engine = create_engine(connection_string)

# Load tables
books = pd.read_sql("SELECT * FROM books", con=engine)
users = pd.read_sql("SELECT * FROM users", con=engine)
interactions = pd.read_sql("SELECT * FROM interactions", con=engine)

def get_recommendations_by_user(user_id, top_n=10):
    try:
        user_id = int(user_id)
    except ValueError:
        raise ValueError("User ID must be an integer.")

    if user_id not in users['user_id'].values:
        raise ValueError(f"User ID {user_id} not found in the database.")

    user_pref = users.loc[users['user_id'] == user_id, 'preferred_genre'].values[0]

    if user_id not in interactions['user_id'].values:
        genre_books = books[books['genres'].str.contains(user_pref, case=False, na=False)]
        if genre_books.empty:
            genre_books = books.sample(top_n)
        return genre_books[['title', 'author', 'coverImg']].head(top_n).to_dict(orient='records')

    candidate_books = books['bookId'].values
    user_array = np.full(len(candidate_books), user_id)
    scores = ncf_model.predict([user_array, candidate_books], verbose=0).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    recommended_book_ids = candidate_books[top_indices]

    genre_filtered = books[books['genres'].str.contains(user_pref.split()[0], case=False, na=False)]
    if not genre_filtered.empty:
        genre_book_ids = set(genre_filtered['bookId'])
        filtered_ids = [bid for bid in recommended_book_ids if bid in genre_book_ids]
        recommended_book_ids = filtered_ids or list(candidate_books[top_indices])

    return books[books['bookId'].isin(recommended_book_ids)][['title', 'author', 'coverImg']].head(top_n).to_dict(orient='records')


import ast

def get_recommendations_by_genre(genre, top_n=10):
    print(f"Genre received: {genre}")

    try:
        # Safely convert stringified lists to actual Python lists
        def parse_genre(g):
            try:
                return ast.literal_eval(g) if isinstance(g, str) and g.startswith('[') else []
            except Exception:
                return []

        books['genres'] = books['genres'].apply(parse_genre)

        # Filter by genre (case-insensitive)
        filtered_books = books[books['genres'].apply(
            lambda g: genre.lower() in [x.lower() for x in g]
        )]

        # If no matches found, return random fallback
        if filtered_books.empty:
            print("No matches found for genre, returning random books.")
            return books.sample(top_n)[['title', 'genres', 'author', 'coverImg']].to_dict(orient='records')

        return filtered_books.head(top_n)[['title', 'author', 'coverImg']].to_dict(orient='records')

    except Exception as e:
        print(f"Error in genre filtering: {e}")
        return {'error': str(e)}
