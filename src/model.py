import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import re
from nltk.stem import WordNetLemmatizer
import openai
import os
import textwrap
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import database

# OpenAI API setup
openai.api_key = os.environ['OPENAI_KEY']
openai.organization = os.environ['OPEN_ORG']

# List of bad and good words
bad_words = {
    "haineux", "déteste", "horrible", "imbécile", "nul", "inutile", "ferme", "stupide", "idiot", "débile", "dégage", "pire",
    "connard", "connasse", "merde", "foutre", "salope", "putain", "bâtard", "enculé", "enculée",
    "hateful", "hate", "horrible", "stupid", "useless", "shut", "idiot", "dumb", "go away", "worse",
    "bastard", "asshole", "shit", "fuck", "bitch", "whore", "prick", "slut", "fool",
}

good_words = {
    "aime", "merci", "bravo", "excellent", "clair", "intéressant", "impressionné", "utile", "super", "adorer",
    "super", "cool", "bien", "belle", "incroyable", "extraordinaire", "excellent", "bravo",
    "love", "thank", "good", "excellent", "clear", "interesting", "impressed", "useful", "great",
    "awesome", "cool", "beautiful", "incredible", "extraordinary", "well done"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def flag_words(text):
    words = clean_text(text)
    return {
        "bad": [word for word in words if word in bad_words],
        "good": [word for word in words if word in good_words]
    }

def analyze_sentiment(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Respond only with 1 or 0. You are a sentiment analysis assistant. Classify text as positive (1) or negative (0)."},
            {"role": "user", "content": f"Text: '{text}'\nIs this positive (1) or negative (0)?"}
        ],
        max_tokens=5
    )
    print(response)
    sentiment = response["choices"][0]["message"]["content"].strip()
    return 1 if sentiment == "1" else 0

def load_data():
    data = database.fetch_tweets()
    print(f"Data loaded: {len(data)} records")
    if not data:
        return pd.DataFrame(columns=["text", "positive", "negative"])
    return pd.DataFrame(data, columns=["text", "positive", "negative"])

def train_model():
    df = load_data()
    if df.empty:
        print("No data available for training.")
        vectorizer = CountVectorizer(stop_words="english")
        model = LogisticRegression()
        vectorizer.fit([""])
        return model, vectorizer

    X = df["text"]
    y = df["positive"]

    flagged_data = [flag_words(text) for text in X]
    y_gpt = [analyze_sentiment(text) for text in X]

    vectorizer = CountVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y_gpt)

    print("Model trained successfully.")
    return model, vectorizer

def generate_confusion_matrix(y_true, y_pred, class_names, filename):
    os.makedirs("src/reports", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    file_path = os.path.join("src/reports", filename)
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved at: {file_path}")

def analyze_model_performance(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Negative", "Positive"])
    generate_confusion_matrix(y_true, y_pred, ["Negative", "Positive"], "positive_confusion_matrix.png")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    return report, {
        "Positive Class": {"Precision": precision[1], "Recall": recall[1], "F1-Score": f1[1]},
        "Negative Class": {"Precision": precision[0], "Recall": recall[0], "F1-Score": f1[0]}
    }

def evaluate_model():
    df = load_data()
    if df.empty:
        print("No data available for evaluation.")
        return

    X = df["text"]
    y = df["positive"]
    model, vectorizer = train_model()
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    report, analysis = analyze_model_performance(y, y_pred)
    pdf_report_path = generate_pdf_report(report, analysis)
    print(f"Evaluation report generated at: {pdf_report_path}")
    return pdf_report_path

def retrain():
    evaluate_model()

def generate_ai_analysis(report, analysis):
    prompt = f"""
    Given the classification report:
    {report}

    And the performance analysis:
    {analysis}

    Provide a detailed AI-generated analysis including observations and recommendations.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

def draw_wrapped_text(c, text, x, y, max_chars=80, line_height=15, bottom_margin=50):
    lines = []
    for original_line in text.splitlines():
        wrapped = textwrap.wrap(original_line, width=max_chars)
        if not wrapped:
            lines.append("")
        else:
            lines.extend(wrapped)

    for line in lines:
        if y < bottom_margin:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750
        c.drawString(x, y, line)
        y -= line_height
    return y

def generate_pdf_report(report, analysis):
    os.makedirs("src/reports", exist_ok=True)
    file_path = "src/reports/evaluation_report.pdf"

    ai_analysis = generate_ai_analysis(report, analysis)

    c = canvas.Canvas(file_path, pagesize=letter)
    c.setFont("Helvetica", 12)

    y_position = 750  # starting y coordinate

    # Title
    y_position = draw_wrapped_text(
        c, "Rapport d’Évaluation du Modèle", 100, y_position, max_chars=50)

    # Space after title
    y_position -= 20

    # Classification Report
    y_position = draw_wrapped_text(
        c, "Classification Report:", 100, y_position, max_chars=80)
    y_position -= 10
    y_position = draw_wrapped_text(c, report, 100, y_position, max_chars=80)

    # Add some space before next section
    y_position -= 20

    # Performance Analysis
    y_position = draw_wrapped_text(
        c, "Performance Analysis:", 100, y_position, max_chars=80)
    y_position -= 10
    # Convert analysis dict into text lines
    analysis_text = ""
    for label, metrics in analysis.items():
        analysis_text += (f"{label}: Precision = {metrics['Precision']:.2f}, "
                          f"Recall = {metrics['Recall']:.2f}, F1-Score = {metrics['F1-Score']:.2f}\n")
    y_position = draw_wrapped_text(
        c, analysis_text, 100, y_position, max_chars=80)

    # Add some space before the AI analysis
    y_position -= 20

    # AI Generated Observations and Recommendations
    y_position = draw_wrapped_text(
        c, "AI-Generated Observations and Recommendations:", 100, y_position, max_chars=80)
    y_position -= 10
    y_position = draw_wrapped_text(
        c, ai_analysis, 100, y_position, max_chars=80)

    # Add the confusion matrix image
    # Check if there is enough space, if not, add a new page.
    if y_position < 250:  # Adjust this threshold as needed for your image height
        c.showPage()
        y_position = 750
    c.drawImage("src/reports/positive_confusion_matrix.png",
                100, y_position - 200, width=400, height=200)

    c.save()
    return file_path

if __name__ == "__main__":
    try:
        train_model()
        evaluate_model()
    except Exception as e:
        print(f"Error during training: {e}")
