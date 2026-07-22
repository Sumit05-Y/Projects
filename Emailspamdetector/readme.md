# 📧 Email Spam Detector

A Machine Learning-based **Email Spam Detection** application that classifies email messages as **Spam** or **Ham (Not Spam)**. The project uses **Natural Language Processing (NLP)** with **TF-IDF Vectorization**, trains multiple classification models, exposes predictions through a **FastAPI** backend, and provides an interactive **Streamlit** frontend.

---

## 📌 Features

- 📧 Detects whether an email is **Spam** or **Ham**
- 🧹 Text preprocessing using **TF-IDF Vectorizer**
- 🤖 Trains and compares multiple Machine Learning models
- ⚡ REST API built with **FastAPI**
- 🎨 Interactive frontend built with **Streamlit**
- 💾 Saves the trained model and vectorizer using **Joblib**

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Random Forest
- FastAPI
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
EmailSpamDetector/
│
├── app.py              # Streamlit frontend
├── main.py             # FastAPI backend
├── train_model.py      # Model training and evaluation
├── data.csv            # Email dataset
├── model.pkl           # Trained machine learning model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── README.md           # Project documentation
└── .venv/              # Virtual environment (ignored in Git)
```

---

## ⚙️ Machine Learning Workflow

1. Load the email dataset.
2. Split the data into training and testing sets.
3. Convert text into numerical features using **TF-IDF Vectorization**.
4. Train multiple classification models:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Random Forest
5. Evaluate each model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
6. Save the best-performing model and vectorizer using Joblib.
7. Serve predictions through FastAPI.
8. Interact with the model using the Streamlit web application.

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics help compare the performance of different classification algorithms.

---

## 🚀 Running the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/EmailSpamDetector.git
cd EmailSpamDetector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

This generates:

- `model.pkl`
- `vectorizer.pkl`

---

### 4️⃣ Start the FastAPI Backend

```bash
python -m uvicorn main:app --reload
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

---

### 5️⃣ Launch the Streamlit Frontend

```bash
python -m streamlit run app.py
```

---

## 📬 API Endpoint

### POST `/predict`

#### Request

```json
{
  "message": "Congratulations! You have won a free iPhone."
}
```

#### Response

```json
{
  "message": "Congratulations! You have won a free iPhone.",
  "prediction": "Spam"
}
```

---

## 🎯 Example Predictions

**Spam**

```
Congratulations!
You have won a free vacation.
Click here to claim your prize.
```

**Ham**

```
Hi,
Can we schedule our meeting for tomorrow at 10 AM?
```

---

## 📈 Models Compared

| Model | Purpose |
|--------|---------|
| Logistic Regression | Text Classification |
| Decision Tree | Classification |
| K-Nearest Neighbors | Classification |
| Random Forest | Ensemble Classification |

---

## 📌 Future Improvements

- Email confidence score
- Probability visualization
- Custom Streamlit UI
- Deployment using Render or Railway
- Docker support
- Email file (.eml) prediction
- Batch email prediction

---

## 👨‍💻 Author

**Sumit Sah**

---

## ⭐ If you found this project helpful, consider giving it a star!