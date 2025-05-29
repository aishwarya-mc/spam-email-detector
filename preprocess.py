import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('stopwords')

def preprocess_text(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def main():
    df = pd.read_csv('data/cleaned_spam.csv')
    df['processed_message'] = df['message'].apply(preprocess_text)
    df.to_csv('data/preprocessed_spam.csv', index=False)
    print("Preprocessing complete. Sample:")
    print(df[['message', 'processed_message']].head())

if __name__ == "__main__":
    main()
