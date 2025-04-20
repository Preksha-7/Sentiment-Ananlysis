import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Start timer
start_time = time.time()

# Load the cleaned data
print("Loading cleaned data...")
data = pd.read_csv('cleaned_data_backup.csv')

# Map sentiment to binary values
data['sentiment_binary'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Features and target
X = data['cleaned_review']
y = data['sentiment_binary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create a pipeline with TF-IDF and Logistic Regression
# Using pre-optimized parameters instead of grid search
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'sentiment_pipeline.pkl')
print("Model saved successfully.")

# Evaluate on test set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Get feature importance
tfidf = pipeline.named_steps['tfidf']
feature_names = tfidf.get_feature_names_out()

coefs = pipeline.named_steps['classifier'].coef_[0]
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefs
})

# Top positive and negative features
top_positive = coef_df.sort_values('Coefficient', ascending=False).head(20)
top_negative = coef_df.sort_values('Coefficient').head(20)

# Plot feature importance
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
sns.barplot(x='Coefficient', y='Feature', data=top_positive, palette='Blues_d')
plt.title('Top Positive Features')
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(x='Coefficient', y='Feature', data=top_negative, palette='Reds_d')
plt.title('Top Negative Features')

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save important features
important_features = pd.concat([top_positive, top_negative])
important_features.to_csv('important_features.csv', index=False)
print("Important features saved to 'important_features.csv'")

# End timer and report
end_time = time.time()
print(f"Model training and evaluation completed in {end_time - start_time:.2f} seconds.")