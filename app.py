import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------
# Path to the saved model pipeline
# ----------------------------------
# If the .pkl is in the same folder as app.py:
MODEL_PATH = Path(__file__).resolve().parent / "Breast_cancer_KNN.pkl"
# If you later move it to models/, change to:
# MODEL_PATH = Path(__file__).resolve().parent / "models" / "knn.joblib"

# ----------------------------------
# Load model once (cached)
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------------
# Figure out which features the model expects
# ----------------------------------
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    # Fallback: manual list (adjust to match how you trained!)
    feature_names = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
    ]

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")
st.title("🔍 Breast Cancer Detection App")
st.markdown(
    """
    This application predicts whether a breast tumor is **benign** or **malignant** 
    based on biopsy features (Wisconsin Breast Cancer Dataset).
    
    ⚠️ **Disclaimer:** This is *not* a medical diagnostic tool. Consult a medical professional for real diagnoses.
    """
)

st.header("Enter Tumor Features")

# Build inputs
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(
        feature.replace('_', ' ').capitalize(),
        min_value=0.0,
        value=1.0,
        format="%.4f"
    )

input_df = pd.DataFrame([user_input])

# Reorder DataFrame columns to match training order exactly (important!)
input_df = input_df[feature_names]

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][prediction]
        else:
            proba = None

        result = "Malignant (High Risk)" if prediction == 1 else "Benign (Low Risk)"
        color = "red" if prediction == 1 else "green"

        st.markdown(
            f"### **Prediction:** <span style='color:{color}'>{result}</span>",
            unsafe_allow_html=True
        )

        if proba is not None:
            st.write(f"**Confidence:** {proba * 100:.2f}%")

        st.info("⚠️ Note: This result is for educational purposes only.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Made with ❤️ using Streamlit & scikit-learn.")

