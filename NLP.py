# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# =========================================
# 2. LOAD DATASET
# =========================================
# Replace 'reviews.csv' with your file path
# Dataset must have: review, sentiment

try:
    data = pd.read_csv("reviews.csv")
except:
    # Sample dataset if file not found
    data = pd.DataFrame({
        "review": [
            "I love this product",
            "Worst purchase ever",
            "Very good quality",
            "Not worth the money",
            "Excellent experience",
            "Terrible service",
            "I am happy with this",
            "I hate it",
            "Amazing product",
            "Bad quality"
        ],
        "sentiment": [
            "positive","negative","positive","negative","positive",
            "negative","positive","negative","positive","negative"
        ]
    })

print("Dataset Preview:\n", data.head())

# =========================================
# 3. TEXT PREPROCESSING
# =========================================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))   # Remove symbols
    text = text.lower()                         # Lowercase
    words = text.split()                        # Tokenize
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned'] = data['review'].apply(preprocess)

print("\nCleaned Text:\n", data[['review', 'cleaned']].head())

# =========================================
# 4. FEATURE EXTRACTION (TF-IDF)
# =========================================
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data['cleaned']).toarray()
y = data['sentiment']

# =========================================
# 5. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 6. MODEL TRAINING (LOGISTIC REGRESSION)
# =========================================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================================
# 7. MODEL EVALUATION
# =========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =========================================
# 8. CONFUSION MATRIX VISUALIZATION
# =========================================
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================================
# 9. CUSTOM SENTIMENT PREDICTION
# =========================================
def predict_sentiment(text):
    processed = preprocess(text)
    vector = tfidf.transform([processed]).toarray()
    result = model.predict(vector)
    return result[0]

# Test examples
print("\nCustom Predictions:")
print("Input: 'This product is amazing' →", predict_sentiment("This product is amazing"))
print("Input: 'Very bad experience' →", predict_sentiment("Very bad experience"))

# =========================================
# 10. SAVE MODEL (OPTIONAL)
# =========================================
import pickle

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
