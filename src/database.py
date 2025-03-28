import mysql.connector
from mysql.connector import Error
import config

def create_connection():
    try:
        connection = mysql.connector.connect(
            host=config.MYSQL_HOST,
            database=config.MYSQL_DATABASE,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def insert_tweet(text, positive, negative):
    connection = create_connection()
    cursor = connection.cursor()
    query = "INSERT INTO tweets (text, positive, negative) VALUES (%s, %s, %s)"
    cursor.execute(query, (text, positive, negative))
    connection.commit()
    cursor.close()
    connection.close()

def fetch_tweets():
    connection = create_connection()
    cursor = connection.cursor()
    query = "SELECT text, positive, negative FROM tweets"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result
