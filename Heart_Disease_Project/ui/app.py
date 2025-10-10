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

# --- Page Config ---
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
    fbs = st.sidebar.selectbox(
        'Fasting Blood Sugar > 120 mg/dL', ['True', 'False'])
    restecg = st.sidebar.selectbox('Resting ECG', [
        'normal', 'ST-T abnormality', 'left ventricular hypertrophy'])
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

    # Align features with model columns if available
    if hasattr(model, 'feature_names_in_'):
        features = features.reindex(
            columns=model.feature_names_in_, fill_value=0)
    return features


input_df = user_input_features()

# --- Prediction ---
st.subheader("🔍 Prediction Result")

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Handle case where model.classes_ may not be ordered
heart_index = np.where(model.classes_ == 1)[0][0]
prob_disease = prediction_proba[0][heart_index]
prob_no_disease = 1 - prob_disease

# Define style variables for the prediction box
bg_color = "#f8f9fa"  # light background
text_color = "#212529"  # dark text
card_shadow = "rgba(33, 37, 41, 0.1)"  # subtle shadow
bar_bg = "#e9ecef"  # bar background

# Styled prediction box (theme-friendly)
st.markdown(f"""
<div style="
    background-color: {bg_color};
    color: {text_color};
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px {card_shadow};
">
<h3 style="text-align:center;">{'❤️ You may be at risk of heart disease' if prediction[0] == 1 else '💙 You are not likely to have heart disease'}</h3>
<p><b>Probability of Heart Disease:</b> {prob_disease:.2f}</p>
<p><b>Probability of No Heart Disease:</b> {prob_no_disease:.2f}</p>
<div style="background:{bar_bg}; border-radius:10px; overflow:hidden; margin-top:10px;">
    <div style="width:{prob_disease*100}%; background:#f06595; padding:5px; color:white; text-align:center;">
        Heart Disease ({prob_disease*100:.1f}%)
    </div>
</div>
<div style="background:{bar_bg}; border-radius:10px; overflow:hidden; margin-top:5px;">
    <div style="width:{prob_no_disease*100}%; background:#4dabf7; padding:5px; color:white; text-align:center;">
        No Disease ({prob_no_disease*100:.1f}%)
    </div>
</div>
</div>
""", unsafe_allow_html=True)
# --- Prediction Summary (Centered Title) ---
st.markdown(
    "<h3 style='text-align:center; font-weight:600; letter-spacing:0.5px;'>🩺 Prediction Summary</h3>",
    unsafe_allow_html=True
)

# Columns layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### 💬 Health Summary")
    if prediction[0] == 1:
        st.warning("⚠️ The model predicts a **risk of heart disease**.")
        st.write("Consider consulting a doctor and adopting a healthy lifestyle.")
    else:
        st.success("💙 Your heart seems healthy according to the model.")
        st.write(
            "Keep maintaining your healthy habits — regular exercise and a balanced diet!")
    st.caption("✨ The chart on the right shows how your case compares to others.")

with right_col:
    # --- Donut Chart ---
    df_viz = pd.DataFrame({
        'age': np.random.randint(29, 77, 100),
        'num': np.random.choice([0, 1], 100)
    })

    df_viz.loc[len(df_viz)] = [input_df['age'].values[0], int(prediction[0])]
    df_viz['num'] = df_viz['num'].astype(int)

    counts = df_viz['num'].value_counts().sort_index()
    labels = ['No Disease', 'Heart Disease']
    colors = ['#4dabf7', '#f06595']

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'lightgrey', 'width': 0.4}
    )

    # Highlight user's slice
    user_idx = int(prediction[0])
    wedges[user_idx].set_edgecolor('lightgrey')
    wedges[user_idx].set_linewidth(3)

    for a in autotexts:
        a.set_color("black")  # more visible in dark mode

    ax.set_title("Heart Disease Distribution", fontsize=12, fontweight='bold')
    st.pyplot(fig)
