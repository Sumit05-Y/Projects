# ✍️Digit Detection Using CNN

A Deep Learning project that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The project includes a **FastAPI backend** for serving the trained model and a **Streamlit frontend** that allows users to upload handwritten digit images and receive predictions in real time.

---

# 📌 About

This project demonstrates an end-to-end deep learning workflow, from training a Convolutional Neural Network (CNN) to deploying it with a simple web interface.

The model is trained on the MNIST handwritten digit dataset and learns to classify grayscale images into one of ten classes (0–9). A FastAPI backend handles model inference, while Streamlit provides an interactive interface for users to upload handwritten digit images and view predictions.

This project was built as part of my journey in learning Deep Learning, Computer Vision, and Model Deployment.

---

# 🚀 Features

- digit recognition (0–9)
- CNN model built using TensorFlow/Keras
- Trained on the MNIST dataset
- Image preprocessing before prediction
- FastAPI REST API backend
- Streamlit frontend
- Real-time prediction
- Prediction confidence score
- Saved and loaded trained model

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow (PIL)
- Matplotlib
- FastAPI
- Uvicorn
- Streamlit
- Requests

---

# 📂 Project Structure

```text
DigitDetector/
│
├── app.py                     # Streamlit Frontend
├── main.py                    # FastAPI Backend
├── cnn_mnist_model.keras      # Trained CNN Model
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project uses the **MNIST Handwritten Digits Dataset** provided by TensorFlow.

Dataset Information:

- 70,000 handwritten digit images
- 60,000 training images
- 10,000 testing images
- Grayscale images
- Image size: **28 × 28**
- Classes: **0–9**

Dataset is loaded directly using:

```python
tf.keras.datasets.mnist.load_data()
```

---

# ⚙️ Data Preprocessing

During training:

- Convert images to float32
- Normalize pixel values (0–255 → 0–1)
- Reshape images to (28,28,1)

During prediction:

- Convert uploaded image to grayscale
- Detect and crop the handwritten digit
- Resize while maintaining aspect ratio
- Center the digit on a 28×28 canvas
- Normalize pixel values
- Pass the processed image to the CNN model

---

# 🧠 CNN Architecture

The model consists of the following layers:

1. Conv2D (32 filters, 3×3, ReLU)
2. MaxPooling2D
3. Conv2D (64 filters, 3×3, ReLU)
4. MaxPooling2D
5. Flatten
6. Dense (128 neurons, ReLU)
7. Dropout (0.5)
8. Dense (10 neurons, Softmax)

---

# 🏋️ Model Training

Training Configuration

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Epochs | 15 |
| Batch Size | 128 |
| Validation Split | 20% |

After training, the model is saved as:

```text
cnn_mnist_model.keras
```

---

# 🌐 API Endpoints

## GET /

Returns the API status.

Response:

```json
{
    "message": "CNN Handwritten Digit Detection API is Running!"
}
```

---

## POST /predict

Accepts an uploaded image and returns:

```json
{
    "prediction": 5,
    "confidence": 98.76
}
```

---

# 💻 Streamlit Frontend

The frontend allows users to:

- Upload a handwritten digit image
- Preview the uploaded image
- Send the image to the FastAPI backend
- Display the predicted digit
- Display the prediction confidence

---

# ▶️ Running the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
cd HandwrittenDigitDetector
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Start the FastAPI Backend

```bash
uvicorn main:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 4. Start the Streamlit Frontend

```bash
streamlit run app.py
```

Frontend URL:

```
http://localhost:8501
```

---

# 📈 Project Workflow

```text
User Uploads Image
          │
          ▼
 Streamlit Frontend
          │
          ▼
 FastAPI Backend
          │
          ▼
 Image Preprocessing
          │
          ▼
 CNN Model Prediction
          │
          ▼
 Predicted Digit + Confidence
          │
          ▼
 Results Displayed to User
```

---

# 📸 Example

1. Upload a handwritten digit image.
2. The image is preprocessed.
3. The CNN predicts the digit.
4. The predicted digit and confidence score are displayed on the screen.

---

# ⚠️ Drawbacks / Limitations

Although the model performs well on the MNIST dataset, it has several limitations:

- The model is trained only on the MNIST dataset and performs best on images that closely resemble MNIST-style handwritten digits.
- Real-world handwritten images may contain shadows, uneven lighting, noise, or large empty borders, which can reduce prediction accuracy.
- The model expects grayscale images with a centered digit on a 28×28 canvas. Images that differ significantly from this format may require additional preprocessing.
- It recognizes only **single handwritten digits (0–9)** and cannot detect multiple digits or handwritten words.
- Prediction accuracy depends heavily on the quality of image preprocessing.
- This application is intended as a learning project and is not designed for production-level Optical Character Recognition (OCR).

---

# 🚀 Future Improvements

Possible enhancements include:

- Add a drawing canvas so users can draw digits directly in the browser.
- Improve image preprocessing for better handling of real-world photographs.
- Train the model on a more diverse handwritten dataset.
- Display prediction probabilities for all digit classes.
- Deploy the application online using Docker and cloud services.
- Extend the model to recognize multiple handwritten digits or complete handwritten text.

---

# 📚 Learning Outcomes

Through this project, I learned:

- Convolutional Neural Networks (CNNs)
- TensorFlow/Keras model development
- Image preprocessing techniques
- Deep Learning model training
- Saving and loading trained models
- Building REST APIs using FastAPI
- Creating interactive interfaces with Streamlit
- Integrating a frontend with a machine learning backend
- Deploying deep learning models for inference

---

# 👨‍💻 Author

**Sumit Sah**

This project was developed as part of my Data Science and Deep Learning learning journey to gain practical experience with CNNs, image classification, and model deployment using FastAPI and Streamlit.