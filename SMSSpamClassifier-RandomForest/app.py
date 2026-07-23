import streamlit as st
import joblib

model = joblib.load("model/spam_classifier.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")


st.title("📱 SMS Spam Classifier")
st.write("Enter an SMS message below to check whether it is Spam or Ham.")


msg = st.text_area("Paste your message here:")


if st.button("Classify"):
    if msg.strip():
        X = tfidf.transform([msg])
        pred = model.predict(X)[0]
        conf = model.predict_proba(X)[0][pred]

        if pred == 1:
            st.error(f"🚨 Spam ({conf:.1%} confidence)")
        else:
            st.success(f"✅ Ham ({conf:.1%} confidence)")
    else:
        st.warning("Please enter a message.")