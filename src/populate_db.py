import database

# Sample tweets
tweets_data = [
    ("I love this new phone!", 1, 0),  # Positive sentiment
    ("I am so tired of this rain.", 0, 1),  # Negative sentiment
    ("This movie was amazing!", 1, 0),
    ("I hate waiting in long lines.", 0, 1),
    ("What a beautiful day outside!", 1, 0),
    ("This coffee tastes terrible.", 0, 1),
    ("I just got a new job, I'm so excited!", 1, 0),
    ("The weather is horrible today.", 0, 1),
    ("I love hiking in the mountains.", 1, 0),
    ("The service at the restaurant was awful.", 0, 1),
]

def populate_database():
    try:
        for tweet, positive, negative in tweets_data:
            database.insert_tweet(tweet, positive, negative)
        print("Database populated successfully!")
    except Exception as e:
        print(f"Error while populating the database: {str(e)}")

if __name__ == "__main__":
    populate_database()
