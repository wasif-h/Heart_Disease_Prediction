import streamlit as st
import pandas as pd
import joblib


scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')
models = joblib.load('heart_model.pkl')


# Set page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f4;
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #d62728;
        font-size: 40px;
        font-weight: bold;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #444;
        text-align: center;
    }
    .footer a {
        color: #d62728;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">❤️ Heart Disease Prediction App</div>', unsafe_allow_html=True)

# Subtitle
st.markdown("### Enter Patient Details:")


# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = models.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")


# Footer
st.markdown("""
<div class="footer">
    <br><hr>
    Developed with ❤️ by <b>Wasif Hossain</b> <br>
    <a href="https://www.linkedin.com/in/wasif-h" target="_blank">🔗 LinkedIn</a> |
    <a href="https://github.com/wasif-h/Heart_Disease_Prediction" target="_blank">📂 GitHub Project</a>
</div>
""", unsafe_allow_html=True)