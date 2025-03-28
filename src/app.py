from flask import Flask, request, jsonify
import model
import database
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import time

app = Flask(__name__)

def analyze_sentiment(tweets):
    try:
        # Train the model using the latest data from the database and predict sentiment
        sentiment_model, vectorizer = model.train_model()  # Train model on the fly
        # Check if the vectorizer is fitted
        if not hasattr(vectorizer, 'vocabulary_'):
            # If the vectorizer is not fitted (this happens in empty DB scenario)
            vectorizer.fit(tweets)  # Fit the vectorizer on the incoming tweets

        # Transform the tweets and make predictions
        X_vec = vectorizer.transform(tweets)
        predictions = sentiment_model.predict(X_vec)

        return predictions
    except Exception as e:
        raise Exception(f"Error during sentiment analysis: {str(e)}")


@app.route('/analyze_sentiment', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        tweets = data['tweets']

        if not tweets:
            return jsonify({"error": "No tweets provided"}), 400

        # Analyze sentiment for incoming tweets
        sentiments = analyze_sentiment(tweets)

        # Convert numpy int64 to int for JSON serialization
        # Ensure all predictions are standard int
        sentiments = [int(s) for s in sentiments]

        # Save tweets with sentiment into DB (Even if DB is empty initially)
        for tweet, sentiment in zip(tweets, sentiments):
            positive = 1 if sentiment == 1 else 0
            negative = 0 if sentiment == 1 else 1
            database.insert_tweet(tweet, positive, negative)

        response = {f"tweet{i+1}": sentiment for i,
                    sentiment in enumerate(sentiments)}
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_all_tweets', methods=['GET'])
def get_all_tweets():
    try:
        tweets = database.fetch_tweets()

        if not tweets:
            return jsonify({"message": "No tweets found in the database."}), 404

        response_data = [{"tweet": tweet[0], "positive": tweet[1],
                          "negative": tweet[2]} for tweet in tweets]
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/retrain_model', methods=['POST'])
def retrain_model_endpoint():
    try:
        retrain_model()  # Manually retrain the model
        return jsonify({"message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/evaluate_model', methods=['POST'])
def evaluate_model_endpoint():
    try:
        pdf_report_path = model.evaluate_model()
        return jsonify({"message": "Model evaluation completed successfully.", "report_path": pdf_report_path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to retrain the model every week
def retrain_model():
    try:
        print("Retraining the model...")
        model.retrain()
        print("Model retrained successfully!")
    except Exception as e:
        print(f"Error retraining model: {e}")


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_model, 'interval',
                      weeks=1)  # Retrain once a week
    scheduler.start()


if __name__ == '__main__':
    start_scheduler()
    app.run(debug=True, port=4242)
