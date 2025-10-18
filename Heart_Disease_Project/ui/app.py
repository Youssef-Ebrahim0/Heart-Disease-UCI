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


# Page config
st.set_page_config(page_title="Heart Disease Prediction",
                   page_icon="❤️", layout="wide")

st.title("Heart Disease Risk Assessment ❤️")

# Disclaimer at the top
st.markdown("""
<div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
    <h4 style="color: #856404; margin: 0 0 8px 0;">⚠️ Disclaimer</h4>
    <p style="color: #856404; margin: 0; font-size: 14px;">
        This project uses a machine learning model to predict the likelihood of heart disease based on input data.
        It is intended for educational and demonstration purposes only and should not be used for medical decisions.
        Always consult a licensed medical professional for proper diagnosis and treatment.
        Predictions may contain errors or inaccuracies.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Detect Streamlit dark mode */
[data-testid="stAppViewContainer"][class*="dark"] .risk-title {
    color: #ffffff !important;  /* White for dark mode */
}
[data-testid="stAppViewContainer"]:not([class*="dark"]) .risk-title {
    color: #000000 !important;  /* Black for light mode */
}
</style>
""", unsafe_allow_html=True)
# Professional CSS styling
st.markdown("""
<style>
/* Main container styling */
.main {
    background-color: #f8f9fa;
}

/* Input section styling */
.input-section {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 30px;
    border: 1px solid #e9ecef;
}

/* Form elements styling */
.stSlider > div > div {
    color: #d9437f;
}
.stSelectbox, .stRadio {
    margin-bottom: 15px;
}

/* Professional button */
.stButton > button {
    background: linear-gradient(135deg, #d9437f 0%, #b33269 100%);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 14px 32px;
    border-radius: 10px;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(217, 67, 127, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(217, 67, 127, 0.4);
}

/* Output section styling */
.output-section {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #e9ecef;
    margin-top: 20px;
}

/* Progress bars */
.probability-bar {
    height: 32px;
    border-radius: 8px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
}
.bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 14px;
}

/* Risk indicator */
.risk-indicator {
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    text-align: center;
    border: 2px solid;
}
.risk-high {
    background: #ffe6f0;
    border-color: #d9437f;
    color: #b33269;
}
.risk-low {
    background: #e6f0ff;
    border-color: #228be6;
    color: #1c7ed6;
}

/* Feature importance if available */
.feature-importance {
    margin-top: 25px;
    padding-top: 20px;
    border-top: 1px solid #e9ecef;
}

/* Section headers */
.section-header {
    font-size: 24px;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e9ecef;
}

/* Risk factors styling */
.risk-factor {
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    border-left: 4px solid;
    background: #f8f9fa;
    height: 100%;
}
.risk-high-item {
    border-left-color: #dc3545;
    background: #f8d7da;
}
.risk-medium-item {
    border-left-color: #ffc107;
    background: #fff3cd;
}
.risk-low-item {
    border-left-color: #28a745;
    background: #d4edda;
}

/* Visualization containers */
.viz-container {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.header("Patient Information")

with st.form(key='heart_disease_form', clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Symptoms")
        sex = st.radio('Sex', ['Male', 'Female'], horizontal=True)
        age = st.slider('Age', 20, 100, 50, help="Patient's age in years")
        cp = st.selectbox('Chest Pain Type', [
            'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'],
            help="Type of chest pain experienced")
        trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 120,
                             help="Resting blood pressure in mmHg")
        chol = st.slider('Cholesterol (mg/dL)', 100, 400, 200,
                         help="Serum cholesterol in mg/dL")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', ['True', 'False'],
                           help="Fasting blood sugar level")

    with col2:
        st.subheader("Medical Measurements")
        restecg = st.selectbox('Resting ECG Results',
                               ['normal', 'ST-T abnormality',
                                   'left ventricular hypertrophy'],
                               help="Resting electrocardiographic results")
        thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150,
                            help="Maximum heart rate achieved during exercise")
        exang = st.selectbox('Exercise Induced Angina', ['True', 'False'],
                             help="Exercise induced chest pain")
        oldpeak = st.slider('ST Depression (mm)', 0.0, 10.0, 1.0, step=0.1,
                            help="ST depression induced by exercise relative to rest")
        slope = st.selectbox('Slope of ST Segment',
                             ['upsloping', 'flat', 'downsloping'],
                             help="Slope of the peak exercise ST segment")
        ca = st.slider('Number of Major Vessels', 0, 3, 0,
                       help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox('Thalassemia',
                            ['normal', 'fixed defect', 'reversable defect'],
                            help="Thalassemia test results")
    # More centered approach
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        submit_button = st.form_submit_button(
            label='🔍 Analyze Heart Disease Risk', use_container_width=True)

# Output Section (only shown after prediction)
if submit_button:
    # Prepare input data
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

    input_df = pd.DataFrame(data, index=[0])
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df.reindex(
            columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
    heart_index = np.where(model.classes_ == 1)[0][0]
    prob_disease = prediction_proba[heart_index]
    prob_no_disease = 1 - prob_disease

    # Display results in separate output section
    st.markdown('<div class="output-section">', unsafe_allow_html=True)
    st.header("Risk Assessment Results")

    # Risk indicator
    if prediction == 1:
        st.markdown(f'''
        <div class="risk-indicator risk-high">
            <h2>🚨 Potential Risk of Heart Disease Detected</h2>
            <p style="font-size: 16px; margin: 10px 0;">Based on the provided information, there is an elevated risk of heart disease.</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="risk-indicator risk-low">
            <h2>✅ Low Risk of Heart Disease</h2>
            <p style="font-size: 16px; margin: 10px 0;">Based on the provided information, the risk of heart disease appears to be low.</p>
        </div>
        ''', unsafe_allow_html=True)

    # Visualization Section - Side by side
    st.subheader("📊 Risk Analysis Dashboard")

    # First row: Risk Visualization & Key Health Metrics side by side
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.subheader("🎯 Risk Probability")

        # Create a pie chart for probability distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Heart Disease', 'No Heart Disease']
        sizes = [prob_disease, prob_no_disease]
        colors = ['#d9437f', '#228be6']
        explode = (0.1, 0) if prediction == 1 else (0, 0.1)

        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops={'fontsize': 12})

        # Enhance the pie chart
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)

        ax.axis('equal')
        plt.title('Heart Disease Probability Distribution',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_col2:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.subheader("📈 Key Health Metrics")

        # Normalize key metrics for visualization
        metrics_data = {
            'Metric': ['Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression'],
            'Value': [
                min(trestbps / 200, 1.0),  # Normalize to 0-1
                min(chol / 400, 1.0),      # Normalize to 0-1
                thalach / 220,             # Normalize to 0-1
                min(oldpeak / 6.0, 1.0)    # Normalize to 0-1
            ],
            # Optimal values (normalized)
            'Optimal Range': [0.6, 0.5, 0.68, 0.0]
        }

        metrics_df = pd.DataFrame(metrics_data)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(metrics_df))

        bars = ax2.barh(x_pos, metrics_df['Value'], color=[
                        '#ff6b6b', '#ffa726', '#42a5f5', '#66bb6a'], alpha=0.7)
        ax2.axvline(x=0.7, color='red', linestyle='--',
                    alpha=0.5, label='High Risk Threshold')
        ax2.axvline(x=0.3, color='green', linestyle='--',
                    alpha=0.5, label='Optimal Range')

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(metrics_df['Metric'])
        ax2.set_xlabel('Normalized Value')
        ax2.set_title('Health Metrics Overview')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{width*100:.0f}%', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # Second row: Risk Factors Analysis - 4 factors side by side, then 4 remaining
    st.subheader("🔍 Detailed Risk Factors Analysis")

    # Analyze individual risk factors
    risk_factors = []

    # Age risk
    if age >= 65:
        risk_factors.append(("High Age Risk", f"Age {age} years", "high", "👴"))
    elif age >= 55:
        risk_factors.append(
            ("Moderate Age Risk", f"Age {age} years", "medium", "👨"))
    else:
        risk_factors.append(("Low Age Risk", f"Age {age} years", "low", "👦"))

    # Blood pressure risk
    if trestbps >= 140:
        risk_factors.append(
            ("High Blood Pressure", f"{trestbps} mmHg", "high", "🩸"))
    elif trestbps >= 130:
        risk_factors.append(
            ("Elevated Blood Pressure", f"{trestbps} mmHg", "medium", "📊"))
    else:
        risk_factors.append(
            ("Normal Blood Pressure", f"{trestbps} mmHg", "low", "✅"))

    # Cholesterol risk
    if chol >= 240:
        risk_factors.append(("High Cholesterol", f"{chol} mg/dL", "high", "🍔"))
    elif chol >= 200:
        risk_factors.append(
            ("Borderline Cholesterol", f"{chol} mg/dL", "medium", "⚠️"))
    else:
        risk_factors.append(
            ("Normal Cholesterol", f"{chol} mg/dL", "low", "🥗"))

    # Heart rate risk
    if thalach < 120:
        risk_factors.append(
            ("Low Max Heart Rate", f"{thalach} bpm", "medium", "🐢"))
    else:
        risk_factors.append(
            ("Good Max Heart Rate", f"{thalach} bpm", "low", "💓"))

    # ST depression risk
    if oldpeak >= 2.0:
        risk_factors.append(("Significant ST Depression",
                            f"{oldpeak} mm", "high", "📉"))
    elif oldpeak >= 1.0:
        risk_factors.append(
            ("Mild ST Depression", f"{oldpeak} mm", "medium", "↘️"))
    else:
        risk_factors.append(
            ("Normal ST Segment", f"{oldpeak} mm", "low", "➡️"))

    # Chest pain risk
    if cp == 'asymptomatic':
        risk_factors.append(
            ("Atypical Chest Pain", "Asymptomatic", "high", "🫀"))
    elif cp in ['atypical angina', 'non-anginal']:
        risk_factors.append(("Chest Pain Present", cp.title(), "medium", "💔"))
    else:
        risk_factors.append(
            ("Typical Chest Pain", "Typical angina", "low", "❤️"))

    # Exercise angina risk
    if exang == 'True':
        risk_factors.append(
            ("Exercise Induced Angina", "Present", "high", "🏃‍♂️"))
    else:
        risk_factors.append(("No Exercise Angina", "Absent", "low", "🚶‍♂️"))

    # Major vessels risk
    if ca >= 2:
        risk_factors.append(("Multiple Vessels Affected",
                            f"{ca} vessels", "high", "🩺"))
    elif ca == 1:
        risk_factors.append(
            ("Single Vessel Affected", f"{ca} vessel", "medium", "🔍"))
    else:
        risk_factors.append(("Clear Vessels", f"{ca} vessels", "low", "👍"))

    # Display risk factors in two rows of 4
    for row in range(0, 2):  # Two rows
        cols = st.columns(4)  # Four columns per row
        for col_idx in range(4):
            factor_idx = row * 4 + col_idx
            if factor_idx < len(risk_factors):
                title, value, level, icon = risk_factors[factor_idx]
                with cols[col_idx]:
                    if level == "high":
                        css_class = "risk-high-item"
                    elif level == "medium":
                        css_class = "risk-medium-item"
                    else:
                        css_class = "risk-low-item"

                    st.markdown(f'''
                    <div class="risk-factor {css_class}">
                        <div style="font-size: 24px; text-align: center; margin-bottom: 8px;">{icon}</div>
                        <div class="risk-title" style="font-weight: 600; margin-bottom: 4px; text-align: center; font-size: 14px;">
                            {title}
                        </div>
                        <div style="font-size: 12px; color: #666; text-align: center;">{value}</div>
                    </div>
                    ''', unsafe_allow_html=True)

    # Additional information
    st.markdown("---")
    st.subheader("💡 Recommendations")

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.write("**Immediate Actions:**")
        if prediction == 1:
            st.write("• 📞 Schedule appointment with cardiologist")
            st.write("• 🏥 Consider additional diagnostic tests")
            st.write("• 📋 Review current medications with doctor")
        else:
            st.write("• ✅ Continue regular health checkups")
            st.write("• 🏃 Maintain healthy lifestyle")
            st.write("• 📊 Monitor risk factors annually")

    with rec_col2:
        st.write("**Lifestyle Suggestions:**")
        st.write("• 🥗 Maintain balanced diet low in saturated fats")
        st.write("• 💪 Regular physical activity (150 mins/week)")
        st.write("• 🚭 Avoid smoking and limit alcohol")
        st.write("• 😊 Manage stress through relaxation techniques")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Show placeholder when no prediction has been made
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: #f8f9fa; border-radius: 12px; margin: 20px 0;">
        <h3 style="color: #6c757d; margin-bottom: 15px;">Ready for Risk Assessment</h3>
        <p style="color: #6c757d; font-size: 16px;">
            Please fill in the patient information above and click <strong>"Analyze Heart Disease Risk"</strong><br>
            to receive a comprehensive heart disease risk assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Final disclaimer at the bottom
st.markdown("""
<div style="background: #f8f9fa; border-radius: 8px; padding: 16px; margin-top: 30px; text-align: center;">
    <p style="color: #6c757d; font-size: 12px; margin: 0;">
        <strong>Important:</strong> This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
        Model accuracy may vary and predictions should not replace professional medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)



