from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data["Message"]
Y = data["Category"]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3
)

# Convert text into numerical features
tf = TfidfVectorizer(
    min_df=1,
    stop_words="english",
    lowercase=True
)

X_train_features = tf.fit_transform(X_train)
X_test_features = tf.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Store results
results = []

# Train and evaluate models
for name, model in models.items():

    # Train model
    model.fit(X_train_features, Y_train)

    # Predictions
    train_pred = model.predict(X_train_features)
    test_pred = model.predict(X_test_features)

    # Evaluation metrics
    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)
    precision = precision_score(Y_test, test_pred)
    recall = recall_score(Y_test, test_pred)
    f1 = f1_score(Y_test, test_pred)

    # Save results
    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(results_df.to_string(index=False))




best_model = LogisticRegression(random_state=42)
best_model.fit(X_train_features, Y_train)

# Save model and vectorizer
joblib.dump(best_model, "model.pkl")
joblib.dump(tf, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")