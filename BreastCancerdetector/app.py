# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🔬", layout="wide")

st.title("🔬 Breast Cancer Classifier")
st.write("Predict whether a cell sample is **malignant** or **benign** using cell measurements powered by an SVM model.")

API_URL = "http://localhost:8000/predict"

st.subheader("Enter Patient Cell Measurements")

# Create 3 tabs or columns to organize the 30 features cleanly on screen
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Mean Features")
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, 13.3)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 18.8)
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, 86.2)
    mean_area = st.slider("Mean Area", 140.0, 2500.0, 551.1)
    mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.16, 0.096)
    mean_compactness = st.slider("Mean Compactness", 0.01, 0.35, 0.092)
    mean_concavity = st.slider("Mean Concavity", 0.0, 0.45, 0.061)
    mean_concave_points = st.slider("Mean Concave Points", 0.0, 0.20, 0.033)
    mean_symmetry = st.slider("Mean Symmetry", 0.10, 0.30, 0.179)
    mean_fractal_dimension = st.slider("Mean Fractal Dimension", 0.04, 0.10, 0.061)

with col2:
    st.markdown("### Standard Error Features")
    radius_error = st.slider("Radius Error", 0.10, 2.90, 0.32)
    texture_error = st.slider("Texture Error", 0.35, 5.00, 1.11)
    perimeter_error = st.slider("Perimeter Error", 0.70, 22.0, 2.29)
    area_error = st.slider("Area Error", 6.0, 540.0, 24.5)
    smoothness_error = st.slider("Smoothness Error", 0.001, 0.030, 0.006)
    compactness_error = st.slider("Compactness Error", 0.002, 0.135, 0.020)
    concavity_error = st.slider("Concavity Error", 0.0, 0.40, 0.026)
    concave_points_error = st.slider("Concave Points Error", 0.0, 0.05, 0.011)
    symmetry_error = st.slider("Symmetry Error", 0.007, 0.080, 0.018)
    fractal_dimension_error = st.slider("Fractal Dimension Error", 0.0008, 0.0300, 0.0032)

with col3:
    st.markdown("### Worst / Largest Features")
    worst_radius = st.slider("Worst Radius", 7.0, 36.0, 14.9)
    worst_texture = st.slider("Worst Texture", 12.0, 50.0, 25.4)
    worst_perimeter = st.slider("Worst Perimeter", 50.0, 250.0, 97.6)
    worst_area = st.slider("Worst Area", 180.0, 4250.0, 686.5)
    worst_smoothness = st.slider("Worst Smoothness", 0.07, 0.22, 0.131)
    worst_compactness = st.slider("Worst Compactness", 0.02, 1.06, 0.212)
    worst_concavity = st.slider("Worst Concavity", 0.0, 1.25, 0.227)
    worst_concave_points = st.slider("Worst Concave Points", 0.0, 0.29, 0.100)
    worst_symmetry = st.slider("Worst Symmetry", 0.15, 0.66, 0.282)
    worst_fractal_dimension = st.slider("Worst Fractal Dimension", 0.05, 0.21, 0.080)

st.markdown("---")

# Prediction Trigger
if st.button("Predict Diagnosis", type="primary", use_container_width=True):
    # Assemble all 30 input parameters in exact expected order
    features = [
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
        mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension,
        radius_error, texture_error, perimeter_error, area_error, smoothness_error,
        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
        worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension
    ]
    
    try:
        response = requests.post(API_URL, json={"features": features})
        
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"].upper()
            confidence = result["confidence"]
            
            if prediction == "MALIGNANT":
                st.error(f"⚠️ **Prediction: {prediction}** | Confidence: {confidence:.1%}")
            else:
                st.success(f"✅ **Prediction: {prediction}** | Confidence: {confidence:.1%}")
                
            st.json(result["probabilities"])
        else:
            st.warning(f"API Error Response: {response.json()}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Make sure `uvicorn main:app --reload` is running on port 8000!")