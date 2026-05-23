import pandas as pd
from collections import Counter
import re

def tokenize(text):
    # Lowercase and split into words
    return re.findall(r'\b\w+\b', text.lower())

def compute_overlap(title1, title2):
    words1 = set(tokenize(title1))
    words2 = set(tokenize(title2))
    return len(words1 & words2)  # count of common words

def find_similar_movies(query_title, csv_path="tmdb_5000_movies.csv", top_k=10):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["title"])  # remove empty titles
    df["overlap_score"] = df["title"].apply(lambda t: compute_overlap(query_title, t))

    top_matches = df[df["title"] != query_title].sort_values(by="overlap_score", ascending=False).head(top_k)
    return top_matches[["title", "overview", "genres"]]

# --- Example Run ---
if __name__ == "__main__":
    query = input("Enter a movie title: ")
    results = find_similar_movies(query)

    print(f"\nTop title-based matches for: '{query}'\n")
    for i, row in results.iterrows():
        print(f"🎬 {row['title']}\n📖 {row['overview']}\n")
