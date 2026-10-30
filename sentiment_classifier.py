from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV         # Hyperparameter tuning (Grid Search)

import pandas as pd

data_path = "dataset/Health_and_Personal_Care.jsonl"

df = pd.read_json(data_path, lines=True)
df = df[["title", "text", "rating"]]

pos_df = df.query("rating == 4 or rating == 5")
neg_df = df.query("rating == 1 or rating == 2")

pos_df["sentiment"] = 1
neg_df["sentiment"] = 0

combined_df = pd.concat([pos_df, neg_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = combined_df['text']
y = combined_df['sentiment']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

svm_model = LinearSVC(random_state=42, C=1.0)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("--- Evaluation Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
