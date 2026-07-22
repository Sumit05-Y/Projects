# verify_reload.py — a brand new file, no relationship to train_model.py
import joblib
from sklearn.datasets import load_breast_cancer

loaded_model = joblib.load("breast_cancer_svm.pkl")
loaded_scaler = joblib.load("breast_cancer_scaler.pkl")

data = load_breast_cancer()
X_scaled = loaded_scaler.transform(data.data)
print(f"Full-dataset accuracy from saved files only: {loaded_model.score(X_scaled, data.target):.3f}")
