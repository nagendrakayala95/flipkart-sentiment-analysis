import pandas as pd
import joblib
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from preprocess import clean_text

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train sentiment analysis model")
parser.add_argument("--data-path", type=str, default=r"C:\Users\LENOVO\Downloads\reviews_data_dump\data.csv",
                    help="Path to CSV file with reviews")
parser.add_argument("--text-col", type=str, default="Review text",
                    help="Name of the column containing review text")
parser.add_argument("--rating-col", type=str, default="Ratings",
                    help="Name of the column containing ratings")

args = parser.parse_args()

# Load Data
data_path = args.data_path
text_col = args.text_col
rating_col = args.rating_col

print(f"Loading data from: {data_path}")
print(f"Text column: {text_col}, Rating column: {rating_col}")

if not os.path.exists(data_path):
    print(f"Data file not found at: {data_path}")
    print("Available columns in current directory:")
    print(os.listdir("."))
    raise FileNotFoundError(f"Data file not found: {data_path}")

try:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    raise RuntimeError(f"Failed to read data file: {e}")

# Verify required columns
required_cols = [text_col, rating_col]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"Available columns: {list(df.columns)}")
    raise KeyError(f"Missing required columns: {missing_cols}. Available: {list(df.columns)}")

# Handle missing values
df = df.dropna(subset=required_cols)

# Create sentiment label
df["sentiment"] = df[rating_col].apply(lambda x: 1 if x >= 4 else 0)

# Clean text
df["cleaned_review"] = df[text_col].fillna("").apply(clean_text)

# Remove rows with empty cleaned reviews
df = df[df["cleaned_review"].str.len() > 0]

# Ensure we still have data
if df.shape[0] == 0:
    raise ValueError("No valid reviews available after cleaning. Check your dataset and preprocessing.")

# Ensure both classes exist
class_counts = df["sentiment"].value_counts()
if class_counts.shape[0] < 2:
    raise ValueError(f"Need at least two classes for training; found: {class_counts.to_dict()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# Save model
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("Model & Vectorizer saved successfully")
