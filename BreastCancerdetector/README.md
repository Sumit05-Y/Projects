# 🔬 Breast Cancer Classification using Support Vector Machine (SVM)

A Machine Learning web application that predicts whether a breast cell sample is **Malignant** or **Benign** using a **Support Vector Machine (SVM)** classifier trained on the Breast Cancer Wisconsin Diagnostic Dataset from Scikit-learn.

The project consists of:

- 🧠 A trained SVM classification model
- ⚡ A FastAPI backend serving predictions
- 🎨 A Streamlit frontend for user interaction
- 💾 Saved model and scaler using Joblib

---

## 📌 Features

- Predicts whether a tumor is **Malignant** or **Benign**
- Interactive Streamlit user interface
- FastAPI REST API for predictions
- Returns prediction confidence
- Displays probability for both classes
- Uses feature scaling with StandardScaler
- Model persistence using Joblib

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- Support Vector Machine (SVM)
- StandardScaler
- FastAPI
- Streamlit
- Joblib
- NumPy
- Requests
- Pydantic

---

## 📂 Project Structure

```
BreastCancer/
│
├── app.py                     # Streamlit frontend
├── main.py                    # FastAPI backend
├── train_model.py             # Train and save the model
├── test.py                    # Verify saved model accuracy
├── breast_cancer_svm.pkl      # Trained SVM model
├── breast_cancer_scaler.pkl   # Saved StandardScaler
└── README.md
```

---

## 📊 Dataset

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset** available in **Scikit-learn**.

It contains:

- 569 samples
- 30 numerical features
- Binary classification

Classes:

- Malignant (Cancerous)
- Benign (Non-cancerous)

Dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

---

## 🧠 Model Training

The model is trained using:

- Support Vector Machine (SVC)
- StandardScaler
- Train-Test Split (80:20)
- Stratified Sampling
- Probability Estimates Enabled

Training pipeline:

1. Load dataset
2. Split training/testing data
3. Scale features
4. Train SVM classifier
5. Evaluate accuracy
6. Save model and scaler

---

## 📈 Model Performance

Example Test Accuracy:

```
Test Accuracy: 98.2%
```

The trained model and scaler are saved using Joblib for deployment.

---

## 🚀 Running the Project

### 1. Install Dependencies

```bash
pip install fastapi uvicorn streamlit scikit-learn numpy joblib requests
```

---

### 2. Train the Model

```bash
python train_model.py
```

This generates:

- breast_cancer_svm.pkl
- breast_cancer_scaler.pkl

---

### 3. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

API runs on:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

### 4. Launch Streamlit

```bash
streamlit run app.py
```

---

## 🔄 Prediction Workflow

```
User Inputs 30 Cell Features
            │
            ▼
      Streamlit Interface
            │
            ▼
     FastAPI REST API
            │
            ▼
   StandardScaler Transform
            │
            ▼
       Trained SVM Model
            │
            ▼
 Prediction + Confidence Score
```

---

## 📥 API Endpoint

### POST `/predict`

Example Request

```json
{
    "features": [
        13.54,
        14.36,
        ...
        0.085
    ]
}
```

Example Response

```json
{
    "prediction": "benign",
    "confidence": 0.992,
    "probabilities": {
        "malignant": 0.008,
        "benign": 0.992
    }
}
```

---

## 🧪 Model Verification

The saved model can be verified independently by running:

```bash
python test.py
```

This loads the saved model and scaler and evaluates the accuracy without retraining.

---

## 🎯 Learning Objectives

This project demonstrates:

- Binary Classification
- Support Vector Machines (SVM)
- Feature Scaling
- Model Serialization
- REST API Development
- FastAPI
- Streamlit Deployment
- Machine Learning Model Serving

---

## 📚 Future Improvements

- Deploy on Streamlit Cloud or Render
- Dockerize the application
- Add input validation
- Improve UI with charts and explanations
- Accept CSV uploads for batch predictions
- Add model monitoring and logging

---

## 👨‍💻 Author

**Sumit Sah**

Aspiring AI & Machine Learning Engineer passionate about building machine learning applications and deploying end-to-end AI solutions.

---

## ⭐ If you found this project helpful, consider giving it a star!