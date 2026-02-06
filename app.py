from flask import Flask, render_template, request, redirect, url_for, session
import os
from movie_recommend import init_feedback_db, DatabaseManager, Embedder, Recommender, save_feedback

app = Flask(__name__)
app.secret_key = "super_secret_key"
db = DatabaseManager("tmdb_5000_movies.csv")
embedder = Embedder()
recommender = Recommender(db, embedder)

@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		session["movie_title"] = request.form["movie_title"]
		session["user_input"] = request.form["user_input"]
		return redirect(url_for('recommend'))
	return render_template("index.html")

@app.route("/recommend")
def recommend():
	movie_title = session.get("movie_title")
	user_input = session.get("user_input")
	if not movie_title or not user_input:
		return redirect(url_for("index"))
	results = recommender.recommend(movie_title, user_input)
	return render_template("results.html", movie_title=movie_title, user_input=user_input, results=results)


@app.route("/feedback", methods=["POST"])
def feedback():
	title = request.form["title"]
	label = request.form["label"]
	overview = request.form["overview"]
	genres = request.form["genres"]
	combined = title + " " + overview
	emb = embedder.encode(combined)
	save_feedback(title, emb, label, genres)
	return "success", 204

@app.route("/update")
def update():
	movie_title = session.get("movie_title")
	user_input = session.get("user_input")
	if not movie_title or not user_input:
		return redirect(url_for("index"))
	results = recommender.recommend(movie_title, user_input)
	session["cached_results"] = [row.to_dict() for _, row in results.iterrows()]
	return redirect(url_for("recommend"))

@app.route("/clear")
def clear_feedback():
	if os.path.exists("feedback.db"):
		os.remove("feedback.db")
	init_feedback_db()
	session.pop("cached_results", None)
	return redirect(url_for("index"))

if __name__ == "__main__":
	init_feedback_db()
	app.run(debug=True)

#http://127.0.0.1:5000/