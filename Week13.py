import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "text": [
        "Breaking news: Aliens landed on Earth",
        "Government announces new policy",
        "Scientists discover new planet",
        "Click here to win a million dollars",
        "The president gave a speech today",
        "Fake cure for cancer discovered",
        "Economy is growing steadily",
        "You won't believe this shocking news"
    ],
    "label": [
        "fake", "real", "real", "fake",
        "real", "fake", "real", "fake"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Test
test = ["Breaking shocking news"]
test_vec = vectorizer.transform(test)
print("\nPrediction:", model.predict(test_vec))