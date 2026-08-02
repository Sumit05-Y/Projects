import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Handwritten Digit Detector")

st.title("✍️ Handwritten Digit Detector")

st.write("Upload an image of a handwritten digit.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files=files
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Predicted Digit: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']}%")

        else:
            st.error("Prediction Failed")