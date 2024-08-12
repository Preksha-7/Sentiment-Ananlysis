from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove punctuations and special characters
    text = text.lower()  # Convert to lower case
    text = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    text = [word for word in text if word not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

@app.route('/')
def home():
    return "Sentiment Analysis Model is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data['review']
        cleaned_review = clean_text(review)
        vectorized_review = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_review)[0]
        sentiment = 'positive' if prediction == 1 else 'negative'
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
