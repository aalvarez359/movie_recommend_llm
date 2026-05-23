import pandas as pd
import torch
import pickle
import sqlite3
import numpy as np
import os
import ast
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

FEEDBACK_DB = 'feedback.db'

def init_feedback_db():
    conn = sqlite3.connect(FEEDBACK_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS global_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_title TEXT,
            embedding BLOB,
            label TEXT,
            genres TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(movie_title, embedding, label, genres):
    conn = sqlite3.connect(FEEDBACK_DB)
    cursor = conn.cursor()
    blob = pickle.dumps(embedding.cpu().numpy().astype(np.float16))
    cursor.execute("""
        INSERT OR REPLACE INTO global_feedback (movie_title, embedding, label, genres, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (movie_title, blob, label, genres, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def load_feedback():
    conn = sqlite3.connect(FEEDBACK_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT embedding, label, genres FROM global_feedback")
    results = cursor.fetchall()
    feedback = []
    for blob, label, genres in results:
        emb = torch.tensor(pickle.loads(blob), dtype=torch.float16)
        feedback.append({'embedding': emb, 'label': label, 'genres': genres})
    conn.close()
    return feedback


def save_embeddings(df, embeddings, db_path='embeddings.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movie_embeddings (
            mov_id INTEGER PRIMARY KEY,
            title TEXT,
            overview TEXT,
            combined_text TEXT,
            genre_words TEXT,
            embedding BLOB
        )
    """)
    for i, row in df.iterrows():
        mov_id = int(row['id'])
        title = row['title']
        overview = row['overview']
        combined_text = row['combined_text']
        genre_words = row['genre_words']
        embedding_blob = pickle.dumps(embeddings[i].cpu().numpy().astype(np.float16))
        cursor.execute("""
            INSERT OR REPLACE INTO movie_embeddings (mov_id, title, overview, combined_text, genre_words, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mov_id, title, overview, combined_text, genre_words, embedding_blob))
    conn.commit()
    conn.close()

def load_embeddings(db_path='embeddings.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT mov_id, title, overview, combined_text, genre_words, embedding FROM movie_embeddings")
    rows = cursor.fetchall()

    ids, titles, overviews, texts, genres, embeddings = [], [], [], [], [], []
    for mov_id, title, overview, text, genre, emb_blob in rows:
        ids.append(mov_id)
        titles.append(title)
        overviews.append(overview)
        texts.append(text)
        genres.append(genre)
        embed_array = pickle.loads(emb_blob).astype(np.float16)
        embeddings.append(torch.tensor(embed_array, dtype=torch.float16))

    conn.close()
    embed_tensor = torch.stack(embeddings)
    df = pd.DataFrame({'id': ids, 'title': titles, 'overview': overviews, 'combined_text': texts, 'genre_words': genres})
    return df, embed_tensor

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    def encode(self, text):
        return self.model.encode([text], convert_to_tensor=True, device='cpu')[0].half()

    def encode_list(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, device='cpu').half()

class DatabaseManager:
    def __init__(self, mov_csv, db_path='embeddings.db'):
        self.mov_csv = mov_csv
        self.db_path = db_path
        self.df = self._prepare_data()
        self.movie_embeddings = self._load_or_create_embeddings()

    def _prepare_data(self):
        df = pd.read_csv(self.mov_csv)
        df['overview'] = df['overview'].fillna("")
        df['genres'] = df['genres'].fillna("[]")
        df['keywords'] = df['keywords'].fillna("[]")

        def json_parser(json_str):
            try:
                items = ast.literal_eval(json_str)
                return " ".join([item['name'] for item in items if 'name' in item])
            except:
                return ""
    
        df['genre_words'] = df['genres'].apply(json_parser)
        df['keyword_words'] = df['keywords'].apply(json_parser)
        df['combined_text'] = df.apply(
            lambda row: f"{row['title']} {row['overview']} {row['genre_words']} {row['keyword_words']}", axis=1
        )
        return df

    def _load_or_create_embeddings(self):
        if os.path.exists(self.db_path):
            df, embeddings = load_embeddings(self.db_path)
            self.df = df
            return embeddings
        else:
            embedder = Embedder()
            print("Encoding all movie embeddings...")
            embeddings = embedder.encode_list(self.df['combined_text'].tolist())
            save_embeddings(self.df, embeddings, self.db_path)
            print("Embeddings saved to database.")
            return embeddings

class Recommender:
    def __init__(self, db: DatabaseManager, embedder: Embedder):
        self.db = db
        self.embedder = embedder

    def recommend(self, movie_title, user_input, top_k=10):
        df = self.db.df
        row = df[df['title'].str.contains(movie_title, case=False, na=False)]

        if row.empty:
            print("❌ Movie not found.")
            return []

        idx = row.index[0]
        query_text = df.iloc[idx]['combined_text'] + " " + user_input
        query_embedding = self.embedder.encode(query_text)

        feedback = load_feedback()
        liked = [f['embedding'] for f in feedback if f['label'] == 'like']
        disliked = [f['embedding'] for f in feedback if f['label'] == 'dislike']

        genre_texts = []
        if 'genre_words' in df.columns:
            genre_texts.append(df.iloc[idx]['genre_words'])
        for f in feedback:
            if f['label'] == 'like' and f['genres']:
                genre_texts.append(f['genres'])

        if genre_texts:
            genre_embedding = self.embedder.encode(" ".join(genre_texts))
            query_embedding += 0.1 * genre_embedding

        cosine_scores = util.cos_sim(query_embedding, self.db.movie_embeddings)[0]

        for emb in liked:
            cosine_scores += 0.3 * util.cos_sim(emb, self.db.movie_embeddings)[0]

        for emb in disliked:
            cosine_scores -= 0.1 * util.cos_sim(emb, self.db.movie_embeddings)[0]

        cosine_scores[idx] = -1

        top_ten = torch.topk(cosine_scores, k=top_k).indices
        return df.iloc[top_ten.tolist()][['title', 'overview', 'genre_words']]


db = DatabaseManager("tmdb_5000_movies.csv")
embedder = Embedder()
recommender = Recommender(db, embedder)

