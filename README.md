# Sentiment Analysis Application

A complete sentiment analysis application demonstrating data preprocessing, model training, and an interactive web interface. Built using IMDB movie reviews data to analyze and predict text sentiment.

## 🚀 Features

- **Data Processing**: Cleans raw text data (stopwords removal, lemmatization, etc.)
- **Exploratory Data Analysis**: Visualizes and summarizes the dataset
- **Feature Engineering**: Uses TF-IDF vectorization for feature extraction
- **Model**: Logistic regression for sentiment prediction
- **Interactive Web Interface**: User-friendly dark-themed frontend for real-time sentiment analysis

## 📁 Repository Structure

```
├── app.py                    # Flask application for the web interface
├── eda.py                    # Exploratory Data Analysis script
├── feature_engineering.py    # Feature engineering logic
├── pre-processing.py         # Data cleaning and preprocessing
├── logistic_regression.py    # Model training script
├── predict.py                # Prediction script for sentiment analysis
├── model_evaluation.py       # Evaluation of model performance
├── static/                   # Static assets for web interface
│   ├── css/
│   │   └── style.css         # CSS styling for dark theme
│   └── js/
│       └── script.js         # Frontend JavaScript
├── templates/                # HTML templates
│   └── index.html            # Main page template
├── sentiment_pipeline.pkl    # Trained sentiment model pipeline
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer for text transformation
├── important_features.csv    # Top features identified by the model
├── README.md                 # Project documentation
└── .gitignore                # Files to ignore in version control
```

## 📊 Dataset

Download the dataset from Kaggle: [IMDB Movie Reviews](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews)

### How to Use the Dataset
1. Download the dataset from the above link
2. Save it in the working directory where the scripts are located

## 🛠️ Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have the trained sentiment model (`sentiment_pipeline.pkl`) in the root directory

## 🚀 Running the Application

### Training the Model

If you want to train the model from scratch:

1. Run the preprocessing script: `python pre-processing.py`
2. Explore the data: `python eda.py`
3. Generate features: `python feature_engineering.py`
4. Train the model: `python logistic_regression.py`
5. Evaluate the model: `python model_evaluation.py`

### Using the Web Interface

To use the interactive web interface:

1. Start the Flask server: `python app.py`
2. Open your web browser and go to `http://127.0.0.1:5000/`
3. Enter or paste text into the input area
4. Click "Analyze Sentiment" or press Ctrl+Enter
5. View the analysis results including sentiment, confidence score, and key factors

## 🌐 Web Interface Features

- Modern dark-themed UI with responsive design
- Real-time sentiment analysis of user input
- Confidence score visualization with a progress bar
- Key factors that influenced the sentiment prediction
- Copy/paste functionality
- Detailed error handling

## 📦 Dependencies

- Flask
- Pandas
- NLTK
- scikit-learn
- joblib
- matplotlib
- seaborn
- numpy
- wordcloud (for EDA)

## 📝 Note

This application uses a machine learning model trained on IMDB movie reviews, but it can analyze various types of text input. The model provides both sentiment classification (positive/negative) and confidence score.
