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
# users_df = pd.read_sql("SELECT * FROM users", con=engine)
# interactions_df = pd.read_sql("SELECT * FROM interactions", con=engine)

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

    # Fetch user details
    user_query = f"SELECT * FROM users WHERE user_id = {user_id}"
    user_df = pd.read_sql(user_query, con=engine)
    if user_df.empty:
        return {'error': f'User ID {user_id} not found'}

    # Load fresh interactions
    interactions_df = pd.read_sql("SELECT * FROM interactions", con=engine)
    user_interactions = interactions_df[interactions_df['user_id'] == user_id]

    # Cold start case
    if user_interactions.empty:
        preferred_genre = user_df['preferred_genres'].iloc[0]
        genre_books = books_df[books_df['genres'].apply(
            lambda g: preferred_genre.lower() in [x.lower() for x in g]
        )].sort_values(by='rating', ascending=False)

        if len(genre_books) < top_n:
            other_books = books_df[~books_df['bookId'].isin(genre_books['bookId'])] \
                .sort_values(by='rating', ascending=False)
            genre_books = pd.concat([genre_books, other_books])

        return genre_books[['title', 'author', 'coverImg']].head(top_n).to_dict(orient='records')

    # Action weights
    action_weights = {
        'click': 1,
        'view': 2,
        'requested': 2,
        'like': 3,
        'search': 0.5
    }

    # Score genres from interactions
    user_interactions['weight'] = user_interactions['action'].map(action_weights).fillna(0)
    merged = pd.merge(user_interactions, books_df, left_on='book_id', right_on='bookId')

    genre_scores = {}
    for _, row in merged.iterrows():
        for genre in row['genres']:
            genre_lower = genre.lower()
            genre_scores[genre_lower] = genre_scores.get(genre_lower, 0) + row['weight']

    if not genre_scores:
        return books_df.sort_values(by='rating', ascending=False)[
            ['title', 'author', 'coverImg']
        ].head(top_n).to_dict(orient='records')

    # Sort genres and get books
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    top_genres = [g for g, _ in sorted_genres[:3]]

    # Filter books from top genres
    filtered_books = books_df[books_df['genres'].apply(
        lambda genres: any(g in [x.lower() for x in genres] for g in top_genres)
    )]

    if filtered_books.empty:
        filtered_books = books_df  # fallback to all books

    # Use NCF model to rank books
    book_ids = filtered_books['bookId'].values
    user_array = np.full(len(book_ids), user_id)
    scores = ncf_model.predict([user_array, book_ids], verbose=0).flatten()

    filtered_books = filtered_books.copy()
    filtered_books['score'] = scores
    recommended_books = filtered_books.sort_values(by='score', ascending=False).head(top_n)

    return recommended_books[['title', 'author', 'coverImg']].to_dict(orient='records')



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
