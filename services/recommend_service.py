import os
import ast
import logging
import warnings
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model

# Suppress TensorFlow and Flask logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
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

# Load tables once at startup
books_df = pd.read_sql("SELECT * FROM books", con=engine)
users_df = pd.read_sql("SELECT * FROM users", con=engine)
interactions_df = pd.read_sql("SELECT * FROM interactions", con=engine)

# Safely parse genres column
def parse_genres(genres_str):
    try:
        return ast.literal_eval(genres_str) if isinstance(genres_str, str) and genres_str.startswith('[') else []
    except Exception:
        return []

books_df['genres'] = books_df['genres'].apply(parse_genres)

def get_recommendations_by_user(user_id, top_n=10):
    try:
        user_id = int(user_id)
    except ValueError:
        return {'error': 'Invalid user_id format'}

    user_query = f"SELECT * FROM users WHERE user_id = {user_id}"
    user_df = pd.read_sql(user_query, con=engine)

    if user_df.empty:
        # Don't recommend if user doesn't exist
        return {'error': f'User ID {user_id} not found'}

    preferred_genre = user_df['preferred_genre'].iloc[0]

    if user_id not in interactions_df['user_id'].values:
        genre_books = books_df[books_df['genres'].apply(
            lambda g: preferred_genre.lower() in [x.lower() for x in g]
        )]

        if genre_books.empty:
            return books_df.sort_values(by='rating', ascending=False)[
                ['title', 'author', 'coverImg']
            ].head(top_n).to_dict(orient='records')

        return genre_books.sort_values(by='rating', ascending=False)[
            ['title', 'author', 'coverImg']
        ].head(top_n).to_dict(orient='records')

    # Predict scores using the model
    candidate_books = books_df['bookId'].values
    user_array = np.full(len(candidate_books), user_id)
    scores = ncf_model.predict([user_array, candidate_books], verbose=0).flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    recommended_book_ids = candidate_books[top_indices]

    # Filter recommended by preferred genre
    genre_filtered_books = books_df[books_df['genres'].apply(
        lambda g: preferred_genre.lower() in [x.lower() for x in g]
    )]
    if not genre_filtered_books.empty:
        genre_book_ids = set(genre_filtered_books['bookId'])
        filtered_ids = [bid for bid in recommended_book_ids if bid in genre_book_ids]
        recommended_book_ids = filtered_ids or list(candidate_books[top_indices])

    return books_df[books_df['bookId'].isin(recommended_book_ids)][
        ['title', 'author', 'coverImg']
    ].head(top_n).to_dict(orient='records')


def get_recommendations_by_genre(genre, top_n=10):
    print(f"Genre received: {genre}")

    try:
        filtered_books = books_df[books_df['genres'].apply(
            lambda g: genre.lower() in [x.lower() for x in g]
        )]

        if filtered_books.empty:
            print("No matches found for genre, returning top-rated books.")
            return books_df.sort_values(by='rating', ascending=False)[
                ['title', 'author', 'coverImg']
            ].head(top_n).to_dict(orient='records')

        return filtered_books.sort_values(by='rating', ascending=False)[
            ['title', 'author', 'coverImg']
        ].head(top_n).to_dict(orient='records')

    except Exception as e:
        print(f"Error in genre filtering: {e}")
        return {'error': str(e)}
