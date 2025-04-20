import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': sentiment_score
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

if __name__ == "__main__":
    # Command line interface
    print("Sentiment Analysis Predictor")
    print("----------------------------")
    
    while True:
        print("\nEnter a review to analyze (or 'q' to quit):")
        text = input("> ")
        
        if text.lower() == 'q':
            break
        
        if not text.strip():
            continue
        
        result = predict_sentiment(text)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Score (0=Negative, 1=Positive): {result['score']:.2f}")