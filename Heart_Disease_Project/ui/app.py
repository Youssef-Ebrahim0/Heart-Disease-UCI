import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Load the saved model pipeline (includes preprocessing)
# Load trained model safely
model_path = os.path.join("models", "heart_disease_rf_pipeline.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)
print("✅ Model loaded successfully!")
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Risk Prediction")

# --- Sidebar Input ---
st.sidebar.header("Enter Your Health Data:")


def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', [
        'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dL)', 100, 400, 200)
    fbs = st.sidebar.selectbox(
        'Fasting Blood Sugar > 120 mg/dL', ['True', 'False'])
    restecg = st.sidebar.selectbox(
        'Resting ECG', ['normal', 'ST-T abnormality', 'left ventricular hypertrophy'])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['True', 'False'])
    oldpeak = st.sidebar.slider(
        'ST depression induced by exercise', 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox('Slope of ST segment', [
                                 'upsloping', 'flat', 'downsloping'])
    ca = st.sidebar.slider(
        'Number of major vessels colored by fluoroscopy', 0, 3, 0)
    thal = st.sidebar.selectbox(
        'Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == 'True' else 0,
        'restecg_1': 1 if restecg == 'ST-T abnormality' else 0,
        'restecg_2': 1 if restecg == 'left ventricular hypertrophy' else 0,
        'thalach': thalach,
        'exang': 1 if exang == 'True' else 0,
        'oldpeak': oldpeak,
        'slope_2': 1 if slope == 'flat' else 0,
        'slope_3': 1 if slope == 'downsloping' else 0,
        'ca': ca,
        'thal_6.0': 1 if thal == 'fixed defect' else 0,
        'thal_7.0': 1 if thal == 'reversable defect' else 0,
        'cp_2': 1 if cp == 'atypical angina' else 0,
        'cp_3': 1 if cp == 'non-anginal' else 0,
        'cp_4': 1 if cp == 'asymptomatic' else 0
    }

    features = pd.DataFrame(data, index=[0])

    # --- Robust feature alignment ---
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        features = features.reindex(columns=model_features, fill_value=0)
    else:
        # fallback: just use the input as-is (assuming model was trained with same column order)
        pass

    return features


input_df = user_input_features()

# --- Prediction ---
st.subheader("Prediction Result")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write("**Predicted Heart Disease:**",
         "Yes ❤️" if prediction[0] == 1 else "No 💙")
st.write("**Prediction Probability:**")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")

# --- Data Visualization (Heart Disease Distribution) ---
st.subheader("Heart Disease Trends (Sample Dataset)")

# Sample dataset
df_viz = pd.DataFrame({
    'age': np.random.randint(29, 77, 100),
    'num': np.random.choice([0, 1], 100)
})

# Smaller, professional-looking figure
fig, ax = plt.subplots(figsize=(6, 4))  # smaller size
sns.countplot(x='num', data=df_viz, ax=ax,
              palette="Set2")  # nicer color palette

# Labels and title
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Disease', 'Heart Disease'], fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_xlabel("Condition", fontsize=12)
ax.set_title("Heart Disease Distribution", fontsize=14, fontweight='bold')

# Remove top and right spines for cleaner look
sns.despine(ax=ax)

# Show in Streamlit
st.pyplot(fig)
