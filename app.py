from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Ensure NLTK data is downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Removing HTML Tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Removing punctuations and special characters
    text = text.lower()  # Converting the entire text to lower case
    text = word_tokenize(text)  # Tokenize the text to words
    # Removing stop words
    text = [word for word in text if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def predict_sentiment(text, model_path='sentiment_pipeline.pkl'):
    """Predict sentiment of a text using the trained model"""
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Make prediction
        sentiment_score = model.predict_proba([cleaned_text])[0, 1]
        sentiment = "positive" if sentiment_score > 0.5 else "negative"
        confidence = max(sentiment_score, 1 - sentiment_score)
        
        # Get top features from the model to explain the prediction
        tfidf = model.named_steps['tfidf']
        classifier = model.named_steps['classifier']
        
        # Get feature names
        feature_names = tfidf.get_feature_names_out()
        
        # Transform the input text
        features = tfidf.transform([cleaned_text])
        
        # Get non-zero feature indices and their values
        non_zero_features = features.nonzero()[1]
        feature_values = features.data
        
        # Get coefficients for these features
        coefficients = classifier.coef_[0]
        
        # Calculate feature contributions (value * coefficient)
        contributions = []
        for idx, value in zip(non_zero_features, feature_values):
            feature_name = feature_names[idx]
            coefficient = coefficients[idx]
            contribution = value * coefficient
            contributions.append((feature_name, contribution))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: -abs(x[1]))
        
        # Get top positive and negative contributions
        top_contributions = contributions[:5]
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': sentiment_score,
            'top_contributions': top_contributions
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text to analyze'})
        
        result = predict_sentiment(text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)