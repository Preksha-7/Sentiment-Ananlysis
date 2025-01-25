import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train_and_save_tfidf_vectorizer():
    data = pd.read_csv('cleaned_data_backup.csv')

    vectorizer = TfidfVectorizer(max_features=10000)
    X = vectorizer.fit_transform(data['cleaned_review'])

    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("TF-IDF vectorizer saved successfully.")

if __name__ == "__main__":
    train_and_save_tfidf_vectorizer()
