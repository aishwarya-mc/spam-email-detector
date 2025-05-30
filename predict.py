import joblib
import re
import sys

model = joblib.load("models/spam_classifier.joblib")  # or use r"models\spam_classifier.joblib"
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your message here\"")
    sys.exit(1)

message = sys.argv[1]

cleaned = clean_text(message)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)

label = "SPAM" if prediction[0] == 1 else "HAM"
print(f"Prediction: {label}")
