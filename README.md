# Sentiment Analysis Application

A complete end-to-end sentiment analysis application featuring data preprocessing, machine learning model training, and an interactive web interface. Built with IMDB movie reviews data to demonstrate real-world text sentiment classification.

## 🚀 Features

- **Data Processing Pipeline**: Comprehensive text cleaning including stopword removal, lemmatization, and text normalization
- **Exploratory Data Analysis**: Rich visualizations and statistical summaries of the dataset
- **Feature Engineering**: TF-IDF vectorization with optimal parameter tuning
- **Machine Learning Model**: Logistic regression classifier with hyperparameter optimization
- **Interactive Web Interface**: Modern, responsive dark-themed frontend for real-time sentiment analysis
- **Model Evaluation**: Comprehensive performance metrics and analysis

## 📁 Repository Structure

```
sentiment-analysis-app/
├── app.py                    # Flask web application
├── eda.py                    # Exploratory Data Analysis
├── feature_engineering.py    # Feature extraction and engineering
├── pre-processing.py         # Data cleaning and preprocessing
├── logistic_regression.py    # Model training and optimization
├── predict.py                # Sentiment prediction utilities
├── model_evaluation.py       # Model performance evaluation
├── static/                   # Web interface assets
│   ├── css/
│   │   └── style.css         # Modern dark theme styling
│   └── js/
│       └── script.js         # Frontend JavaScript functionality
├── templates/                # HTML templates
│   └── index.html            # Main application interface
├── models/                   # Trained model artifacts
│   ├── sentiment_pipeline.pkl    # Complete trained pipeline
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── data/                     # Dataset directory
├── results/                  # Model outputs and analysis
│   └── important_features.csv    # Top predictive features
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore               # Version control ignore rules
```

## 📊 Dataset

This project uses the **IMDB Movie Reviews Dataset** from Kaggle, containing 50,000 movie reviews labeled as positive or negative.

**Download**: [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews)

### Dataset Setup
1. Download the dataset from the Kaggle link above
2. Create a `data/` directory in the project root
3. Extract the dataset files into the `data/` directory

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if not already present)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```

## 🚀 Usage

### Option 1: Quick Start (Pre-trained Model)

If you have the pre-trained model files:

```bash
python app.py
```

Open your browser to `http://127.0.0.1:5000/` and start analyzing text sentiment!

### Option 2: Train from Scratch

To train the model from scratch:

```bash
# 1. Preprocess the data
python pre-processing.py

# 2. Explore the dataset
python eda.py

# 3. Engineer features
python feature_engineering.py

# 4. Train the model
python logistic_regression.py

# 5. Evaluate performance
python model_evaluation.py

# 6. Launch the web app
python app.py
```

## 🌐 Web Interface

The application features a modern, responsive web interface with:

- **Dark Theme**: Easy on the eyes for extended use
- **Real-time Analysis**: Instant sentiment prediction as you type
- **Confidence Scoring**: Visual progress bar showing prediction confidence
- **Key Factors**: Display of words that most influenced the prediction
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Keyboard Shortcuts**: Ctrl+Enter for quick analysis
- **Error Handling**: Graceful handling of edge cases and errors

### Interface Features
- Text input area with placeholder guidance
- One-click sentiment analysis
- Confidence score visualization
- Top influencing words display
- Copy/paste functionality
- Loading states and animations

## 📈 Model Performance

The logistic regression model achieves:
- **Accuracy**: ~89% on test data
- **Precision**: High precision for both positive and negative classes
- **Recall**: Balanced recall across sentiment classes
- **F1-Score**: Strong F1 scores indicating good overall performance

## 🔧 Dependencies

### Core Libraries
```
flask>=2.0.0
pandas>=1.3.0
nltk>=3.6
scikit-learn>=1.0.0
joblib>=1.1.0
numpy>=1.21.0
```

### Visualization & Analysis
```
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

### Web Interface
```
gunicorn>=20.1.0  # For production deployment
```

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## 📝 API Usage

The application also supports programmatic access:

```python
from predict import predict_sentiment

# Analyze text sentiment
text = "This movie was absolutely amazing!"
result = predict_sentiment(text)
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Technical Details

### Model Architecture
- **Algorithm**: Logistic Regression with L2 regularization
- **Vectorization**: TF-IDF with n-gram range (1,2)
- **Preprocessing**: Stopword removal, lemmatization, lowercasing
- **Features**: Top 10,000 most important features selected

### Performance Metrics
- Cross-validation accuracy: 89.2% ± 0.8%
- Training time: ~2 minutes on standard hardware
- Prediction time: <100ms per text sample

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**: Download required NLTK data
   ```python
   import nltk
   nltk.download('all')
   ```

3. **Model Files Missing**: Ensure model files are in the correct directory
   ```bash
   ls models/  # Should show .pkl files
   ```

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code documentation

---

**Note**: This application uses a machine learning model trained specifically on movie reviews, but it can effectively analyze sentiment in various types of text input. The model provides both sentiment classification (positive/negative) and confidence scoring for transparency.
