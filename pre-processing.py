import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
print("NLTK data downloaded.")

# Load dataset - fix path issue
dataset_path = 'IMDB Dataset.csv'
if not os.path.exists(dataset_path):
    print(f"WARNING: {dataset_path} not found.")
    dataset_path = input("Please enter the correct path to the IMDB dataset: ")

print(f"Loading dataset from {dataset_path}...")
try:
    data = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
    print(data.head())
    print(f"Dataset shape: {data.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

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

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(data)}")
print(f"Sentiment distribution:\n{data['sentiment'].value_counts()}")
print(f"Average review length (characters): {data['review'].str.len().mean():.2f}")
print(f"Average cleaned review length (characters): {data['cleaned_review'].str.len().mean():.2f}")