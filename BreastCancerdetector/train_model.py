from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# probability=True is required to get confidence scores later, not just labels
model = SVC(random_state=42, probability=True)
model.fit(X_train_scaled, y_train)

print(f"Test accuracy: {model.score(X_test_scaled, y_test):.3f}")
# Test accuracy: 0.982 — matches Day 11 exactly

joblib.dump(model, "breast_cancer_svm.pkl")
joblib.dump(scaler, "breast_cancer_scaler.pkl")
print("Saved model and scaler.")