import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Road Accident Severity Prediction", page_icon="🚦", layout="wide")
st.title("🚦 Road Accident Severity Prediction")
st.caption("Machine Learning mini-project | Educational demonstration")

MODEL_PATH = Path("accident_severity_model.joblib")
if not MODEL_PATH.exists():
    st.warning("Model file not found. Train the model first using: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter accident and road conditions")
c1, c2, c3 = st.columns(3)
with c1:
    road_type = st.selectbox("Road type", ["Urban", "Highway", "Rural", "Intersection"])
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy"])
    light = st.selectbox("Lighting", ["Daylight", "Dark - street lights", "Dark - no street lights"])
with c2:
    traffic = st.selectbox("Traffic density", ["Low", "Medium", "High"])
    surface = st.selectbox("Road surface", ["Dry", "Wet", "Icy", "Gravel"])
    speed = st.slider("Speed limit (km/h)", 20, 120, 60)
with c3:
    vehicles = st.slider("Vehicles involved", 1, 6, 2)
    driver_age = st.slider("Driver age", 18, 85, 35)
    alcohol = st.selectbox("Alcohol involvement", ["No", "Yes"])

if st.button("Predict accident severity", type="primary"):
    sample = pd.DataFrame([{
        "road_type": road_type, "weather": weather, "light": light,
        "traffic_density": traffic, "road_surface": surface,
        "speed_limit": speed, "vehicles_involved": vehicles,
        "driver_age": driver_age, "alcohol_involvement": alcohol
    }])
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.classes_
    confidence = float(max(probabilities)) * 100

    st.divider()
    st.subheader("Prediction result")
    if prediction == "Severe":
        st.error(f"Predicted severity: {prediction}")
    elif prediction == "Moderate":
        st.warning(f"Predicted severity: {prediction}")
    else:
        st.success(f"Predicted severity: {prediction}")
    st.metric("Model confidence", f"{confidence:.1f}%")
    st.caption("Class probabilities")
    st.bar_chart(pd.DataFrame({"Probability (%)": probabilities * 100}, index=classes))

st.info("This is an academic demonstration using a synthetic sample dataset. It is not an official road-safety assessment tool and must not be used for emergency or legal decisions.")
