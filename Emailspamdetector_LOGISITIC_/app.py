import streamlit as st
import requests

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detector")

st.write("Enter an email message below to check whether it is **Spam** or **Ham**.")

message = st.text_area("Email Message", height=200)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter an email message.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"message": message}
            )

            prediction = response.json()["prediction"]

            if prediction == "Spam":
                st.error("🚨 This email is SPAM.")
            else:
                st.success("✅ This email is HAM.")

        except Exception:
            st.error("Cannot connect to FastAPI server.")




 #python -m streamlit run app.py