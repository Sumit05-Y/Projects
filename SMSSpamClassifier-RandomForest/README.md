# 📱 SMS Spam Classifier (Random Forest)

A Machine Learning-based **SMS Spam Classification** project that classifies text messages as **Spam** or **Ham (Not Spam)**. The project applies **Natural Language Processing (NLP)** techniques using **TF-IDF Vectorization** for feature extraction and a **Random Forest Classifier** for prediction. The trained model is deployed through an interactive **Streamlit** web application, allowing users to classify SMS messages in real time.

> **Note:** I have previously implemented a spam detection project using **Logistic Regression** on an email dataset. This project is a separate implementation that uses a **Random Forest Classifier** on an SMS dataset to explore a different machine learning approach for spam classification.

---

# 📌 Features

- 📱 Classifies SMS messages as Spam or Ham
- 📝 TF-IDF Vectorization for text feature extraction
- 🌲 Random Forest Classifier for prediction
- 📊 Compares Naive Bayes and Random Forest models
- 💾 Saves the trained model and vectorizer using Joblib
- 🌐 Interactive Streamlit web application
- ⚡ Real-time predictions with confidence scores

---

# 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

---

# 📂 Dataset

This project uses the **SMS Spam Collection Dataset**, which contains over **5,500 SMS messages** labeled as either:

- **Ham** (Legitimate Message)
- **Spam** (Unwanted Promotional or Fraudulent Message)

The dataset is cleaned by removing duplicate messages before training the model.

---

# 🔄 Project Workflow

### 1. Data Loading & Exploration

- Load the SMS Spam Collection dataset
- Remove duplicate messages
- Explore class distribution

### 2. Feature Engineering

- Convert labels into numerical values
- Split the dataset into training and testing sets
- Apply **TF-IDF Vectorization** to transform SMS messages into numerical features

### 3. Model Training

Two machine learning models were trained and compared:

- Naive Bayes
- Random Forest Classifier

After evaluation, the **Random Forest Classifier** was selected as the final model for deployment.

### 4. Model Evaluation

The models are evaluated using:

- Classification Report
- Precision
- Recall
- F1-Score

### 5. Deployment

The trained Random Forest model and TF-IDF vectorizer are saved using **Joblib** and deployed through a **Streamlit** application for real-time SMS spam prediction.

---

# 📁 Project Structure

```text
SMSSpamClassifier-RandomForest/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── model/
    ├── spam_classifier.pkl
    └── tfidf_vectorizer.pkl
```

---

# 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
```

### 2️⃣ Navigate to the Project Directory

```bash
cd SMSSpamClassifier-RandomForest
```

### 3️⃣ Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

### 4️⃣ Activate the Virtual Environment

**Windows (Git Bash)**

```bash
source .venv/Scripts/activate
```

### 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The Streamlit application will automatically open in your default web browser.

---

# 🧠 Machine Learning Model

| Component | Technique |
|-----------|-----------|
| Problem Type | Binary Classification |
| Feature Extraction | TF-IDF Vectorization |
| Main Model | Random Forest Classifier |
| Comparison Model | Naive Bayes |
| Model Storage | Joblib |

---

# 📈 Results

- Successfully classifies SMS messages into **Spam** and **Ham**
- Compared the performance of **Naive Bayes** and **Random Forest**
- Selected **Random Forest** as the final deployed model
- Provides confidence scores for each prediction
- Demonstrates the application of NLP techniques for text classification

---

# 🎯 Sample Predictions

### Spam

```
Congratulations!

You have won a FREE iPhone.

Click here to claim your prize now.
```

### Ham

```
Hey, are we still meeting at 6 PM today?
```

---

# 👨‍💻 Author

**Sumit Sah**

Aspiring AI & Machine Learning Engineer passionate about Machine Learning, Data Science, Natural Language Processing, and end-to-end AI application development.

GitHub: https://github.com/Sumit05-Y

---

## ⭐ If you found this project helpful, consider giving it a star!