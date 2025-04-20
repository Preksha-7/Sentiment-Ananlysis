import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve

# Load the model and test data
model = joblib.load('sentiment_pipeline.pkl')
data = pd.read_csv('cleaned_data_backup.csv')

# Map sentiment to binary values
data['sentiment_binary'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data (using the same seed as in training)
from sklearn.model_selection import train_test_split
X = data['cleaned_review']
y = data['sentiment_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
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

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy')
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

# Plot learning curve
print("Generating learning curve (this may take some time)...")
try:
    plot_learning_curve(model, X, y)
    print("Learning curve generated.")
except Exception as e:
    print(f"Error generating learning curve: {e}")

# Error analysis
print("\nPerforming error analysis...")
test_data = pd.DataFrame({
    'text': X_test,
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_prob
})

# Find misclassified examples
misclassified = test_data[test_data['actual'] != test_data['predicted']]
misclassified['confidence'] = misclassified['probability'].apply(lambda x: max(x, 1-x))
misclassified = misclassified.sort_values('confidence', ascending=False)

# Save misclassified examples
misclassified.to_csv('misclassified_examples.csv', index=False)
print(f"Saved {len(misclassified)} misclassified examples to 'misclassified_examples.csv'")

# Print some examples of high-confidence errors
print("\nHigh-confidence misclassifications:")
for _, row in misclassified.head(5).iterrows():
    actual = "Positive" if row['actual'] == 1 else "Negative"
    predicted = "Positive" if row['predicted'] == 1 else "Negative"
    print(f"Text: {row['text'][:100]}...")
    print(f"Actual: {actual}, Predicted: {predicted}, Confidence: {row['confidence']:.2f}")
    print("-" * 80)

print("Evaluation complete.")