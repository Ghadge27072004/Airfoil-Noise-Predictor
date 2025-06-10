# ✈️ Airfoil Noise ML

This is a **Streamlit app** that predicts **Sound Pressure Level (SPL)** based on airfoil measurements using a trained **Random Forest Regressor**. It also provides data visualization and model performance reporting.

---

## 🚀 Features

- **Predictor**: Input features like Frequency, Angle of Attack, Chord Length, etc., and get the predicted SPL.
- **Data & Graphs**: Preview dataset, visualize feature distributions, and explore correlation heatmaps.
- **Model Report**: Check model performance metrics like RMSE and R² score.

---

## 📦 Files Included

- `app.py`: Streamlit app code.
- `best_model.pkl`: Trained Random Forest Regressor model.
- `scaler.pkl`: Feature scaler for preprocessing.
- `airfoil_self_noise.dat`: Dataset file.

---

## 📊 Dataset Info

The dataset includes:
- **Frequency (Hz)**
- **Angle of attack (degrees)**
- **Chord length (m)**
- **Free stream velocity (m/s)**
- **Suction side displacement thickness (m)**
- **Sound pressure level (dB)**

---

## 🛠️ Installation

1. **Clone the repo** or download the files.

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
