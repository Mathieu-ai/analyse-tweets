import re
import os
import textwrap
import database

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
    print('todo train')

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
    print('todo analyse perf')

def evaluate_model():
    print('todo evaluate model')

def retrain():
    print('todo evaluate')
    # TODO evaluate_model()

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
        print('todo train and evaluate')
        # TODO train_model()
        # TODO evaluate_model()
    except Exception as e:
        print(f"Error during training: {e}")
