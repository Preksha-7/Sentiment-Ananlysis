import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load dataset
dataset_path = 'Sentiment-Ananlysis\IMDB Dataset.csv'
print("Loading dataset...")
data = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print(data.head())

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
print("NLTK data downloaded.")

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Removing HTML Tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Removing punctuations and special characters
    text = text.lower()  # Converting the entire text to lower case
    text = word_tokenize(text)  # Tokenize the text to words
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

print("Cleaning reviews...")
data['cleaned_review'] = data['review'].apply(clean_text)
print("Reviews cleaned.")
print(data[['review', 'cleaned_review']].head())

# Save cleaned data
cleaned_data_path = 'cleaned_data_backup.csv'

print(f"Saving cleaned data to {cleaned_data_path}...")
data.to_csv(cleaned_data_path, index=False)
print("Cleaned data saved successfully.")
# Assuming 'sentiment' is the column with labels
print(data['sentiment'].unique())
