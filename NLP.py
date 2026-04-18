# =========================================
# SENTIMENT ANALYSIS USING NLP
# TF-IDF + LOGISTIC REGRESSION
# =========================================

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords', quiet=True)

# =========================================
# 1. LOAD DATASET
# =========================================
try:
    data = pd.read_csv("reviews.csv")
    data.columns = data.columns.str.lower()

    if 'review' not in data.columns or 'sentiment' not in data.columns:
        raise Exception("Dataset must contain review and sentiment columns")

except:
    # Sample dataset
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
            "Bad quality",
            "Fantastic item",
            "Very disappointing"
        ],
        "sentiment": [
            "positive","negative","positive","negative",
            "positive","negative","positive","negative",
            "positive","negative","positive","negative"
        ]
    })

print("Dataset Loaded Successfully")
print(data.head())

# =========================================
# 2. PREPROCESS TEXT
# =========================================
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned_review'] = data['review'].apply(preprocess)

# =========================================
# 3. CONVERT LABELS
# =========================================
data['sentiment'] = data['sentiment'].map({
    'positive': 1,
    'negative': 0
})

data.dropna(inplace=True)

# =========================================
# 4. TF-IDF VECTORIZATION
# =========================================
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(data['cleaned_review'])
y = data['sentiment']

# =========================================
# 5. TRAIN TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# 6. TRAIN MODEL
# =========================================
model = LogisticRegression(
    max_iter=300,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# =========================================
# 7. PREDICTIONS
# =========================================
y_pred = model.predict(X_test)

# =========================================
# 8. EVALUATION
# =========================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(cm)

# =========================================
# 9. VISUALIZATION
# =========================================
plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Negative','Positive'],
    yticklabels=['Negative','Positive']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================================
# 10. CUSTOM PREDICTION
# =========================================
def predict_sentiment(text):
    cleaned = preprocess(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"

# Example
print("\nCustom Predictions:")
print("This product is amazing ->", predict_sentiment("This product is amazing"))
print("Very bad service ->", predict_sentiment("Very bad service"))

# =========================================
# 11. SAVE MODEL
# =========================================
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel saved successfully.")
