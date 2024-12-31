import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

cleaned_data_path = 'D:/Sentiment analysis/cleaned_data.csv'
data = pd.read_csv(cleaned_data_path)

X = data['cleaned_review']
y = data['sentiment']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

count_vectorizer = CountVectorizer(max_features=5000)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(max_features = 5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

data['review_length'] = data['cleaned_review'].apply(len)
data['word_count'] = data['cleaned_review'].apply(lambda x: len(x.split()))

positive_words = ['good', 'great', 'excellent', 'amazing']
negative_words = ['bad', 'terrible', 'awful', 'worst']

def contains_positive_words(text):
    return any(word in text for word in positive_words)
def contains_negative_words(text):
    return any(word in text for word in negative_words)

data['contains_positive_word'] = data['cleaned_review'].apply(contains_positive_words).astype(int)
data['contains_negative_word'] = data['cleaned_review'].apply(contains_negative_words).astype(int)

features_path = 'D:/Sentiment analysis/features.csv'
data.to_csv(features_path, index=False)

print("Feature engineering completed and saved to 'features.csv'.")
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')