import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "review": [
        "This movie is amazing and wonderful",
        "I love this film so much",
        "Excellent acting and great story",
        "Very bad movie and boring",
        "Terrible film I hate it",
        "Worst acting ever",
        "Fantastic movie with great visuals",
        "Awful and disappointing film",
        "I really enjoyed this movie",
        "Poor storyline and bad ending"
    ],
    "sentiment": [
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_review = ["This movie was fantastic and enjoyable"]
new_vec = vectorizer.transform(new_review)
prediction = model.predict(new_vec)

print("\nNew Review Prediction:", prediction[0])