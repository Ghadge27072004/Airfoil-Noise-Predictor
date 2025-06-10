import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
col_names = ["Frequency", "Angle_of_attack", "Chord_length", "Free_stream_velocity", "Suction_side_displacement_thickness", "Sound_pressure_level"]
df = pd.read_csv("airfoil_self_noise.dat", sep="\t", header=None, names=col_names)

# Streamlit Page Config
st.set_page_config(page_title="Airfoil Noise ML", page_icon="✈️", layout="wide")

# App Title
st.markdown("# ✈️ Airfoil Noise ML")

# Simple Navbar using selectbox
page = st.sidebar.selectbox("Menu", ["🏠 Home", "🔍 Predictor", "📊 Data & Graphs", "📈 Model Report"])

# Home
if page == "🏠 Home":
    st.subheader("Welcome")
    st.write("""
        🔸 Predict **Sound Pressure Level (SPL)** using machine learning.  
        🔸 Analyze airfoil data with interactive graphs.  
        🔸 Fast, easy, and accurate predictions!
    """)

# Predictor
elif page == "🔍 Predictor":
    st.subheader("Predict Sound Pressure Level")
    freq = st.number_input("Frequency (Hz)", min_value=0.0)
    angle = st.number_input("Angle of Attack (°)", min_value=0.0)
    chord = st.number_input("Chord Length (m)", min_value=0.0)
    velocity = st.number_input("Free Stream Velocity (m/s)", min_value=0.0)
    thickness = st.number_input("Suction Side Thickness (m)", min_value=0.0)

    if st.button("Predict"):
        X = scaler.transform([[freq, angle, chord, velocity, thickness]])
        prediction = model.predict(X)[0]
        st.success(f"Predicted SPL: {prediction:.2f} dB")

# Data & Graphs
elif page == "📊 Data & Graphs":
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Feature Analysis")
    feature = st.selectbox("Select feature:", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, color="skyblue", ax=ax)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2, use_container_width=True)

# Model Report
elif page == "📈 Model Report":
    st.subheader("Model Performance")
    st.markdown("""
    - **Model:** Random Forest Regressor  
    - **RMSE:** ~1.83  
    - **R² Score:** ~0.93
    """)
