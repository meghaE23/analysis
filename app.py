import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load models and vectorizers with correct paths
sentiment_model = joblib.load("sentiment_model.pkl")

sentiment_vectorizer = joblib.load("sentiment_vectorizer.pkl")

sarcasm_model = joblib.load("sarcasm_model.pkl")
sarcasm_vectorizer = joblib.load("sarcasm_vectorizer.pkl")  # <-- fixed the missing )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    # Sentiment prediction
    sentiment_features = sentiment_vectorizer.transform([review])
    sentiment_pred = sentiment_model.predict(sentiment_features)[0]
    sentiment_label = "Positive" if sentiment_pred == 1 else "Negative"

    # Sarcasm prediction
    sarcasm_features = sarcasm_vectorizer.transform([review])
    sarcasm_pred = sarcasm_model.predict(sarcasm_features)[0]
    sarcasm_label = "Sarcastic" if sarcasm_pred == 1 else "Not Sarcastic"

    return render_template("index.html",
                           sentiment_result=sentiment_label,
                           sarcasm_result=sarcasm_label)

if __name__ == "__main__":
    app.run(debug=True)
