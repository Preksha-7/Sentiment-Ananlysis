# Sentiment-Ananlysis

This repository demonstrates a sentiment analysis pipeline using IMDB movie reviews. It involves data preprocessing, exploratory data analysis, feature engineering, and model training.

## Repository Structure
├── eda.py # Exploratory Data Analysis script
├── feature_engineering.py # Feature engineering logic
├── pre-processing.py # Data cleaning and preprocessing 
├── logistic_regression_model.pkl # Trained logistic regression model 
├── tfidf_vectorizer.pkl # TF-IDF vectorizer for text transformation 
├── README.md # Project documentation 
└── .gitattributes # File attribute settings

## Features

- Preprocessing: Cleans raw text data (stopwords removal, lemmatization, etc.).
- Exploratory Data Analysis: Visualizes and summarizes the dataset.
- Feature Engineering: Uses TF-IDF vectorization for feature extraction.
- Model: Logistic regression for sentiment prediction.

## Dataset

Download the dataset from Kaggle: [IMDB Movie Reviews](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews).

### How to Use the Dataset
1. Download the dataset from the above link.
2. Save it in the working directory where the scripts are located.
