# 📧 Spam Email Detector

A  machine learning model that detects whether a given message is **spam** or **ham (not spam)**.

---

## 🚀 Features

- 📚 Trained on real SMS spam dataset
- 🧠 Uses TF-IDF Vectorizer + Multinomial Naive Bayes
- ⚙️ Preprocessing pipeline for clean inputs
- 📦 Predict directly via command line
- ✅ Easy to retrain & extend

---

## 🗂️ Project Structure

spam-detector/
│
├── data/
│ ├── spam.csv
│ └── cleaned_spam.csv
│ └── preprocessed_spam.csv
│
├── models/
│ ├── spam_classifier.joblib
│ └── tfidf_vectorizer.joblib
│
├── preprocess.py
├── model_train.py
├── predict.py
├── requirements.txt
│──clean.py
└── README.md

### Example Predictions

```bash
$ python predict.py "Win a free iPhone now!"
Prediction: SPAM

$ python predict.py "Hey, how are you doing today?"
Prediction: HAM


