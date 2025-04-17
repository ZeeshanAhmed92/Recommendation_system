import os
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

def get_recommendations(user_id, top_n=10):
    if user_id not in users['user_id'].values:
        return books.sample(top_n)['title'].tolist()

    user_pref = users.loc[users['user_id'] == user_id, 'preferred_genre'].values[0]

    if user_id not in interactions['user_id'].values:
        genre_books = books[books['genres'].str.contains(user_pref, case=False, na=False)]
        if genre_books.empty:
            genre_books = books.sample(top_n)
        return genre_books['title'].head(top_n).tolist()

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

    return books[books['bookId'].isin(recommended_book_ids)]['title'].head(top_n).tolist()
