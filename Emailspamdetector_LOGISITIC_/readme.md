# 📧 Email Spam Detector using Logistic Regression

A Machine Learning-based **Email Spam Detection** application that classifies email messages as **Spam** or **Ham (Not Spam)** using **Logistic Regression**. The project applies **Natural Language Processing (NLP)** with **TF-IDF Vectorization**, serves predictions through a **FastAPI** backend, and provides an interactive **Streamlit** frontend.

This project demonstrates the implementation of Email Spam Detection using **Logistic Regression**. Additional versions of this project using different machine learning classifiers will be developed separately to compare their performance and understand their strengths and limitations.

---

## 📌 Features

- 📧 Classifies emails as **Spam** or **Ham**
- 🧹 Text preprocessing using **TF-IDF Vectorization**
- 🤖 Trains an Email Spam Detection model using **Logistic Regression**
- 📊 Evaluates model performance using multiple classification metrics
- ⚡ REST API built with **FastAPI**
- 🎨 Interactive web interface built with **Streamlit**
- 💾 Saves the trained model and TF-IDF vectorizer using **Joblib**

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
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
├── train_model.py      # Model training
├── data.csv            # Email dataset
├── model.pkl           # Trained Logistic Regression model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── README.md           # Project documentation
└── .venv/              # Virtual environment (ignored in Git)
```

---

## ⚙️ Machine Learning Workflow

1. Load the email dataset.
2. Split the dataset into training and testing sets.
3. Convert email text into numerical features using **TF-IDF Vectorization**.
4. Train a **Logistic Regression** classifier.
5. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
6. Save the trained model and vectorizer using **Joblib**.
7. Deploy the prediction API using **FastAPI**.
8. Interact with the model through the **Streamlit** web application.

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

These metrics help measure the effectiveness of the classifier in distinguishing spam emails from legitimate emails.

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

### Request

```json
{
  "message": "Congratulations! You have won a free iPhone."
}
```

### Response

```json
{
  "message": "Congratulations! You have won a free iPhone.",
  "prediction": "Spam"
}
```

---

## 🎯 Example Predictions

### Spam

```
Congratulations!

You have won a free vacation.
Click here to claim your prize.
```

### Ham

```
Hi,

Can we schedule our meeting for tomorrow at 10 AM?
```

---

## 📈 Model Used

| Model | Task |
|--------|------|
| Logistic Regression | Email Spam Classification |

---

## 📌 Future Implementations

This project is part of a series exploring different machine learning algorithms for Email Spam Detection. Future implementations will include:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)

Each implementation will be developed as a separate project to compare performance, accuracy, and real-world applicability.

---

## 👨‍💻 Author

**Sumit Sah**

---

## ⭐ If you found this project helpful, consider giving it a star