import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import joblib

print("Loading cleaned data...")
cleaned_data_path = 'cleaned_data_backup.csv'
data = pd.read_csv(cleaned_data_path)

print(f"Dataset shape: {data.shape}")

# Basic text features
print("Creating basic text features...")
data['review_length'] = data['cleaned_review'].apply(len)
data['word_count'] = data['cleaned_review'].apply(lambda x: len(x.split()))
data['avg_word_length'] = data['cleaned_review'].apply(lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0)
data['unique_word_count'] = data['cleaned_review'].apply(lambda x: len(set(x.split())))
data['unique_ratio'] = data['unique_word_count'] / data['word_count']
data['uppercase_count'] = data['review'].apply(lambda x: sum(1 for c in x if c.isupper()))
data['uppercase_ratio'] = data['uppercase_count'] / data['review'].apply(len)

# Lexical features
print("Creating lexical features...")
positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'wonderful', 'fantastic', 'best', 'beautiful', 'perfect', 'enjoyed', 'favorite', 'recommend']
negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'disappointing', 'poor', 'boring', 'waste', 'horrible', 'stupid', 'disappointing', 'mediocre']

def count_word_occurrences(text, word_list):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    return sum(word in word_list for word in words)

data['positive_word_count'] = data['cleaned_review'].apply(lambda x: count_word_occurrences(x, positive_words))
data['negative_word_count'] = data['cleaned_review'].apply(lambda x: count_word_occurrences(x, negative_words))
data['positive_negative_ratio'] = data['positive_word_count'] / (data['negative_word_count'] + 1)  # +1 to avoid division by zero

# NLTK Sentiment Analysis
try:
    print("Adding NLTK sentiment scores...")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    # Apply VADER sentiment analysis
    data['vader_compound'] = data['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
    data['vader_pos'] = data['review'].apply(lambda x: sia.polarity_scores(x)['pos'])
    data['vader_neg'] = data['review'].apply(lambda x: sia.polarity_scores(x)['neg'])
    data['vader_neu'] = data['review'].apply(lambda x: sia.polarity_scores(x)['neu'])
except Exception as e:
    print(f"Error adding NLTK sentiment scores: {e}")

# TF-IDF Vectorization
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_review'])
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Visualize key features
print("Creating visualizations...")
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(data=data, x='review_length', hue='sentiment', kde=True, bins=30)
plt.title('Review Length by Sentiment')
plt.xlabel('Review Length (characters)')

plt.subplot(2, 2, 2)
sns.histplot(data=data, x='word_count', hue='sentiment', kde=True, bins=30)
plt.title('Word Count by Sentiment')
plt.xlabel('Word Count')

plt.subplot(2, 2, 3)
sns.histplot(data=data, x='positive_word_count', hue='sentiment', kde=True, bins=20)
plt.title('Positive Word Count by Sentiment')
plt.xlabel('Positive Word Count')

plt.subplot(2, 2, 4)
sns.histplot(data=data, x='negative_word_count', hue='sentiment', kde=True, bins=20)
plt.title('Negative Word Count by Sentiment')
plt.xlabel('Negative Word Count')

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Correlation heatmap of numerical features
numeric_cols = ['review_length', 'word_count', 'avg_word_length', 'unique_word_count', 
                'unique_ratio', 'uppercase_count', 'uppercase_ratio',
                'positive_word_count', 'negative_word_count', 'positive_negative_ratio']

if 'vader_compound' in data.columns:
    numeric_cols += ['vader_compound', 'vader_pos', 'vader_neg', 'vader_neu']

# Add sentiment as binary
data['sentiment_binary'] = data['sentiment'].map({'positive': 1, 'negative': 0})
numeric_cols.append('sentiment_binary')

# Create correlation matrix
corr_matrix = data[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.close()

# Save engineered features
features_path = 'features.csv'
data.to_csv(features_path, index=False)
print(f"Feature engineering completed and saved to '{features_path}'.")

# Feature importance analysis
from sklearn.ensemble import RandomForestClassifier

print("\nCalculating feature importance...")
X = data[numeric_cols[:-1]]  # All except sentiment_binary
y = data['sentiment_binary']

try:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('numeric_feature_importance.png')
    plt.close()
    
    print("Top important features:")
    print(feature_importance.head(10))
except Exception as e:
    print(f"Error in feature importance analysis: {e}")

print("Feature engineering analysis complete.")