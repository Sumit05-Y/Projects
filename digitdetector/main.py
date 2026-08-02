from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Handwritten Digit Detection API")

# Load trained model
model = tf.keras.models.load_model("cnn_mnist_model.keras")




def preprocess_image(image):
    # Convert PIL image to grayscale numpy array
    image = image.convert("L")
    img = np.array(image)

    # Invert colors (black digit on white paper -> white digit on black)
    img = 255 - img

    # Threshold to remove gray pixels
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Find the digit
    coords = cv2.findNonZero(img)

    if coords is None:
        # No digit found
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)

    # Crop digit
    img = img[y:y+h, x:x+w]

    # Keep aspect ratio
    target_size = 20

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    img = cv2.resize(img, (new_w, new_h))

    # Create 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset+new_h,
           x_offset:x_offset+new_w] = img

    # Normalize
    canvas = canvas.astype("float32") / 255.0

    # Optional: Display what the CNN sees
    import matplotlib.pyplot as plt

    plt.imshow(canvas, cmap="gray")
    plt.title("Image sent to CNN")
    plt.axis("off")
    plt.show()

    return canvas.reshape(1, 28, 28, 1)


@app.get("/")
def home():
    return {"message": "CNN Handwritten Digit Detection API is Running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    processed = preprocess_image(image)

    prediction = model.predict(processed, verbose=0)[0]

    print("\nPrediction probabilities:")

    for i, p in enumerate(prediction):
        print(f"{i}: {p:.4f}")

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "prediction": digit,
        "confidence": round(confidence * 100, 2)
    }