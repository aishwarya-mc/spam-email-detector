import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os


df = pd.read_csv("data/preprocessed_spam.csv")
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])  # 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/spam_classifier.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
