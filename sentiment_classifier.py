import pandas as pd

# Separate and Label Classes
data_path = "dataset/Health_and_Personal_Care.jsonl"

# Clean dataset
df = pd.read_json(data_path, lines=True)
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df.query("verified_purchase == True")
df = df[["text", "rating"]]

pos_df = df.query("rating == 4 or rating == 5")
neg_df = df.query("rating == 1 or rating == 2")

pos_df["sentiment"] = 1
neg_df["sentiment"] = 0

# Combine and Shuffle Rows
combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = combined_df['text']
y = combined_df['sentiment']

from sklearn.model_selection import train_test_split

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Create Pipeline for Linear SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('svm', LinearSVC(random_state=42, dual=False))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("--- Evaluation Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'],digits=4))