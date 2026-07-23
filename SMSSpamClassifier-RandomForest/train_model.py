import pandas as pd
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
print(df.shape)                     
print(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)   
print(df["label"].value_counts(normalize=True))

df["target"] = (df["label"] == "spam").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
)

tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_t = tfidf.fit_transform(X_train)   
X_test_t  = tfidf.transform(X_test)


for name, model in [("Naive Bayes", MultinomialNB()),
                    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
    t0 = time.time()
    model.fit(X_train_t, y_train)
    print(name, f"trained in {time.time()-t0:.3f}s")
    print(classification_report(y_test, model.predict(X_test_t), digits=3))

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(tfidf,    "model/tfidf_vectorizer.pkl")