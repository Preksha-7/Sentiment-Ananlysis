import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data = {
    'review': ['good movie', 'bad movie', 'great film', 'terrible film', 'fantastic', 'awful'],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}
df = pd.DataFrame(data)


df['cleaned_review'] = df['review']  
X = df['cleaned_review']
y = df['sentiment']

y = y.map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

log_reg = LogisticRegression()
log_reg.fit(X_train_vec, y_train)

joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'count_vectorizer.pkl')

X_test_vec = vectorizer.transform(X_test)
y_pred = log_reg.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Model and vectorizer saved successfully.")
