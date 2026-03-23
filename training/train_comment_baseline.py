import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

df = pd.read_csv("data/processed/actionable_comments.csv")

train_df = df[df["set"] == "train"].copy()
test_df = df[df["set"] == "test"].copy()

X_train = train_df["Text"]
y_train = train_df["Label2"]

X_test = test_df["Text"]
y_test = test_df["Label2"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", LinearSVC())
])

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Comment Type Classification Report:\n")
print(classification_report(y_test, preds, zero_division=0))