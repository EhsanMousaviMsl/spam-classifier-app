# backend/train/train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("./data/spam.csv", encoding="ISO-8859-1")
df = df[["v1", "v2"]]  # Only keep the relevant columns
df.columns = ["label", "message"]

# 2. Convert labels to binary
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# 4. Create pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# 5. Train the model
pipeline.fit(X_train, y_train)

# 6. Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save the model
joblib.dump(pipeline, "../models/spam_classifier.pkl")
print("✅ Model saved to models/spam_classifier.pkl")
