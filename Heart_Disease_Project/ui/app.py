import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# Load the saved model pipeline (includes preprocessing)
# get absolute path relative to app.py
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(BASE_DIR, "../models/heart_disease_rf_pipeline.pkl"))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Risk Prediction")

# --- Sidebar Input ---
st.sidebar.header("Enter Your Health Data")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', [
        'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol (mg/dL)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dL', ['True', 'False'])
    restecg = st.sidebar.selectbox('Resting ECG', ['normal', 'ST-T abnormality', 'left ventricular hypertrophy'])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['True', 'False'])
    oldpeak = st.sidebar.slider('ST depression induced by exercise', 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox('Slope of ST segment', ['upsloping', 'flat', 'downsloping'])
    ca = st.sidebar.slider('Number of major vessels colored by fluoroscopy', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversable defect'])

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

    # Feature alignment if model has feature names
    if hasattr(model, 'feature_names_in_'):
        features = features.reindex(columns=model.feature_names_in_, fill_value=0)
    return features

input_df = user_input_features()

# --- Prediction ---
st.subheader("🔍 Prediction Result")
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Ensure probability index matches heart disease class
heart_index = np.where(model.classes_ == 1)[0][0]
prob_disease = prediction_proba[0][heart_index]
prob_no_disease = 1 - prob_disease

# Styled prediction card
st.markdown(f"""
<div style="background-color:#f8f9fa; padding:20px; border-radius:15px; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
<h3>{'❤️ You may be at risk of heart disease' if prediction[0]==1 else '💙 You are not likely to have heart disease'}</h3>
<p><b>Probability of Heart Disease:</b> {prob_disease:.2f}</p>
<p><b>Probability of No Heart Disease:</b> {prob_no_disease:.2f}</p>
<div style="background:#f1f3f5; border-radius:10px; overflow:hidden; margin-top:10px;">
    <div style="width:{prob_disease*100}%; background:#f06595; padding:5px; color:white;">Heart Disease</div>
</div>
<div style="background:#f1f3f5; border-radius:10px; overflow:hidden; margin-top:5px;">
    <div style="width:{prob_no_disease*100}%; background:#4dabf7; padding:5px; color:white;">No Disease</div>
</div>
</div>
""", unsafe_allow_html=True)

# --- Heart Disease Distribution ---
st.subheader("📊 Heart Disease Trends (Sample Dataset)")

# Sample dataset
df_viz = pd.DataFrame({
    'age': np.random.randint(29, 77, 100),
    'num': np.random.choice([0, 1], 100)
})

# Highlight user input in distribution
df_viz.loc[len(df_viz)] = [input_df['age'].values[0], prediction[0]]

# Plot
fig, ax = plt.subplots(figsize=(7,5))
sns.countplot(x='num', data=df_viz, palette={0:'#4dabf7', 1:'#f06595'}, ax=ax)

# Labels
ax.set_xticks([0,1])
ax.set_xticklabels(['No Disease', 'Heart Disease'], fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_xlabel("Condition", fontsize=12)
ax.set_title("Heart Disease Distribution", fontsize=14, fontweight='bold')
sns.despine(ax=ax)

st.pyplot(fig)
st.markdown("💡 *Your input is highlighted in the chart above.*")


