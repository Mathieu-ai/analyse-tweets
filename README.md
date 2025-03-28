# Social Sentiment Analysis API

This project provides an API for analyzing sentiments in social media posts (tweets). It allows users to submit tweets, analyze their sentiment (positive or negative), and retrieve the sentiment analysis results. Additionally, the project includes functionality to periodically retrain the model and evaluate its performance.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [API Endpoints](#api-endpoints)
4. [Database Setup](#database-setup)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Scheduled Retraining](#scheduled-retraining)
8. [Populating the Database](#populating-the-database)
9. [Docker Setup](#docker-setup)

---

## Project Overview

The application performs sentiment analysis on tweets to classify them as either positive or negative. It leverages a pre-trained sentiment analysis model from Hugging Face, along with custom logic for processing tweets and storing the results in a MySQL database. The system automatically retrains the model on a regular basis (once every week) and generates evaluation reports to assess the model's performance.

### Key Features

- Sentiment analysis of user-submitted tweets (positive or negative).
- Retraining of the model on a periodic basis.
- Evaluation reports (including a confusion matrix and performance metrics).
- API endpoints for interacting with the system.

---

## Setup Instructions

To get started with the project, follow the steps below.

### Prerequisites

1. **Python 3.x** installed on your local machine.
2. **MySQL** installed and running locally (or via Docker).
3. **pip** for installing Python dependencies.

### Steps to Set Up the Project

1. **Clone the repository**:

    ```bash
    git clone https://gitlab.com/EFREI_MATHIEU/algo/exos/05.03.2025
    cd 05.03.2025
    ```

2. **Install Python dependencies**:
    It is recommended to use a virtual environment to isolate dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. **Set up the database**:
   - Either set up MySQL on your local machine or use the provided Docker setup.
   - If using Docker, skip to the **Docker Setup** section. Otherwise, follow these steps:
     - Create a new MySQL database `socialmetrics`.
     - Update your `src/config.py` file with the correct database credentials.

4. **Run the application**:

    ```bash
    python src/app.py
    ```

---

## API Endpoints

This API provides several endpoints for interacting with the sentiment analysis system:

### 1. **Analyze Sentiment**

- **Endpoint**: `/analyze_sentiment`
- **Method**: `POST`
- **Description**: Accepts a list of tweets and returns their sentiment (positive or negative).
- **Request Body**:

    ```json
    {
        "tweets": [
            "I love this new phone!",
            "This movie was terrible!"
        ]
    }
    ```

- **Response**:

    ```json
    {
        "tweet1": 1,
        "tweet2": 0
    }
    ```

  - `1` = Positive sentiment
  - `0` = Negative sentiment

---

### 2. **Get All Tweets**

- **Endpoint**: `/get_all_tweets`
- **Method**: `GET`
- **Description**: Fetches all tweets stored in the database along with their sentiment classification (positive/negative).
- **Response**:

    ```json
    [
        {"tweet": "I love this new phone!", "positive": 1, "negative": 0},
        {"tweet": "I hate this weather.", "positive": 0, "negative": 1}
    ]
    ```

---

### 3. **Retrain Model**

- **Endpoint**: `/retrain_model`
- **Method**: `POST`
- **Description**: Manually retrains the sentiment analysis model.
- **Response**:

    ```json
    {
        "message": "Model retrained successfully."
    }
    ```

---

### 4. **Evaluate Model**

- **Endpoint**: `/evaluate_model`
- **Method**: `POST`
- **Description**: Evaluates the model's performance and generates a PDF report with the evaluation results.
- **Response**:

    ```json
    {
        "message": "Model evaluation completed successfully.",
        "report_path": "src/reports/evaluation_report.pdf"
    }
    ```

---

## Database Setup

The application requires a MySQL database to store tweets and their sentiment classification. The database schema is as follows:

### Database Table: `tweets`

| Column   | Type     | Description                      |
|----------|----------|----------------------------------|
| text     | TEXT     | The tweet text                   |
| positive | INT      | Sentiment classification (1 = positive, 0 = negative) |
| negative | INT      | Sentiment classification (1 = negative, 0 = positive) |

### Creating the Database

You can set up the database using the following SQL command:

```sql
CREATE DATABASE socialmetrics;

USE socialmetrics;

CREATE TABLE tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT NOT NULL,
    positive INT NOT NULL,
    negative INT NOT NULL
);
```

---

## Model Training

The sentiment analysis model is a logistic regression classifier trained on tweet data stored in the MySQL database. It uses the `CountVectorizer` to vectorize tweet text and the `LogisticRegression` model for classification.

- **Training the Model**: The model is trained using the labeled data stored in the database.
- **Retraining the Model**: You can manually trigger retraining by calling the `/retrain_model` API endpoint or set up automatic retraining using a scheduler (`APScheduler`).

---

## Model Evaluation

Once the model is trained, you can evaluate its performance using the `/evaluate_model` API endpoint. This will generate a PDF report that includes:

- A **classification report** showing precision, recall, and F1-score for both positive and negative sentiments.
- A **confusion matrix** image that visualizes the performance of the model on the test data.

The generated PDF will be saved in the `src/reports` directory.

---

## Scheduled Retraining

To ensure the model stays up to date, the application includes a background scheduler (`APScheduler`) that automatically retrains the model once a week. This helps keep the model aligned with any changes in the sentiment trends of new tweets.

---

## Populating the Database

For testing purposes, the `src/populate_db.py` script populates the database with some sample tweets and their sentiment labels (positive or negative). To populate the database:

1. Run the script:

    ```bash
    python src/populate_db.py
    ```

2. This will insert sample tweets into the `tweets` table.

---

## Docker Setup

To run the application in a containerized environment, you can use Docker. The `docker-compose.yml` file defines the services for running the MySQL database.

### Steps to Set Up Docker

1. **Start the MySQL container**:

    ```bash
    docker-compose up -d
    ```

2. **Verify the MySQL container is running**:

    ```bash
    docker ps
    ```

The MySQL server will be available on port `3306`.

## Demo

Video
[![Demo Video](https://imgs.search.brave.com/Id-q-Jk-cvXWTJi9qs0108jKpnlNEVb8Jg70oN9tovQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5pc3RvY2twaG90/by5jb20vaWQvMTM2/NTMyODAxMi92ZWN0/b3IvdmlkZW8tbWVk/aWEtcGxheWVyLW9u/LWxhcHRvcC1zY3Jl/ZW4uanBnP3M9NjEy/eDYxMiZ3PTAmaz0y/MCZjPXNGYzBJSXJX/dWlzQVhKek9BaC1R/dVViZ2l3bDNDR0Rp/cUxiUkVtcGZaeFE9)](screen/video.mp4)
