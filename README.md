					Contextual Movie Recommendation System


🔁 1. App Initialization (app.py)
When app.py is run:

movie_recommend.py initializes the DatabaseManager, Embedder, and Recommender.

All movies from tmdb_5000_movies.csv are loaded and embedded using SentenceTransformer.

These embeddings are stored in a SQLite database (embeddings.db).

feedback.db is initialized to track likes/dislikes.

🎬 2. User Inputs Movie & Description
You type in:

A movie title (e.g., “Scooby-Doo”)

A sentence describing what you like (e.g., “I like spooky but silly movies.”)

Flask captures that form data and redirects to the /recommend route.

🤖 3. Recommendations Are Generated (Recommender.recommend())
In movie_recommend.py, the recommend() method:

Finds the matching movie row from the dataframe.

Combines that movie’s combined_text (title + overview + genre) with your user input.

Encodes this combo into a query_embedding using the embedder.

Loads all saved feedback (likes/dislikes) from feedback.db.

If likes exist: computes the average liked embedding and adds it to the query.

If dislikes exist: computes the average disliked embedding and subtracts it from the query.

Computes cosine similarity between the query and all movie embeddings.

Sets the input movie’s score to -1 so it’s not recommended again (cosine_scores[idx] = -1).

Picks the top 10 most similar movies.

Returns their titles, overviews, and genre info to be displayed on the HTML page.

👍👎 4. You Click "Like" or "Dislike"
A JavaScript function sends a POST request to the /feedback route.

Flask extracts the movie title, overview, genres, and your label (like/dislike).

The embedding for that movie is re-encoded and saved to feedback.db.

🔁 5. You Click "Update List"
Triggers the /update route, which:

Reruns Recommender.recommend() using your original inputs.

Now includes the updated feedback in its embedding adjustment.

The top 10 list is recalculated and rendered again.
