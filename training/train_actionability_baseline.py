import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/processed/processed_data.csv") if False else pd.read_csv("data/raw/processed_data.csv")

train_df = df[df["set"] == "train"].copy()
test_df = df[df["set"] == "test"].copy()

X_train = train_df["Text"]
y_train = train_df["Label1"]

X_test = test_df["Text"]
y_test = test_df["Label1"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Actionability Classification Report:\n")
print(classification_report(y_test, preds, zero_division=0))