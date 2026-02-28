import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD MODELS ----------------
heart_model = pickle.load(open('heart_disease_prediction.pkl','rb'))
heart_scaler = pickle.load(open('heart_scaler.pkl','rb'))

diabetes_model = pickle.load(open('diabetes_disease_prediction.pkl','rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl','rb'))

# ---------------- RISK FUNCTION ----------------
def get_risk(prob):
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# ---------------- HEART RECOMMENDATION ----------------
def heart_recommendation(risk):

    if risk == "HIGH":
        recommend = [
            "30 min daily walking",
            "Avoid fried food",
            "Reduce salt intake",
            "7–8 hours sleep",
            "Morning yoga",
            "Stress management",
            "Avoid smoking & alcohol",
            "Regular BP monitoring"
        ]

    elif risk == "MEDIUM":
        recommend = [
            "20–30 min brisk walking",
            "Limit junk food",
            "Reduce sugar intake",
            "Sleep at least 7 hours",
            "Light yoga / meditation",
            "Maintain healthy weight",
            "Drink more water"
        ]

    else:
        recommend = [
            "Maintain diet",
            "Continue physical activity",
            "Regular health checkup"
        ]

    st.subheader("Lifestyle Recommendations:")
    for i in recommend:
        st.write(f"- {i}")

# ---------------- DIABETES RECOMMENDATION ----------------
def diabetes_recommendation(risk):

    if risk in ["HIGH", "MEDIUM"]:
        recommend = [
            "Consult with a doctor",
            "Monitor blood sugar levels regularly",
            "Follow low sugar diet",
            "Daily walking",
            "Maintain healthy weight",
            "Sleep 7–8 hours"
        ]
    else:
        recommend = [
            "Maintain diet",
            "Continue physical activity",
            "Regular health checkup"
        ]

    st.subheader("Lifestyle Recommendations:")
    for i in recommend:
        st.write(f"- {i}")

# ---------------- UI ----------------
st.title("🩺 Multi Disease Prediction System")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Heart Disease Prediction", "Diabetes Prediction"]
)

# ================= HEART ====================
if disease == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age",min_value=1,max_value=120,help="Enter your age in years (Normal: 18–80)")
        trestbps = st.number_input( "Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, help="Normal: 90–120 | High: >140")
        chol = st.number_input( "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, help="Normal: <200 | Borderline: 200–239 | High: >240" )
        thalach = st.number_input( "Maximum Heart Rate Achieved", min_value=60, max_value=220, help="Normal: 60–100 (resting)" )
        oldpeak = st.number_input( "ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1, help="Range: 0–6 | Higher = more risk" )

    with col2:

        sex_display = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex_display == "Male" else 0

        cp_display = st.selectbox("Chest Pain Type",
        ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
        cp = ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"].index(cp_display)

        fbs_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"])
        fbs = 1 if fbs_display == "Yes" else 0

        exang_display = st.selectbox("Exercise Induced Angina", ["No","Yes"])
        exang = 1 if exang_display == "Yes" else 0

    with col3:

        restecg_display = st.selectbox("Rest ECG",
        ["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
        restecg = ["Normal","ST-T wave abnormality","Left ventricular hypertrophy"].index(restecg_display)

        slope_display = st.selectbox("Slope",
        ["Upsloping","Flat","Downsloping"])
        slope = ["Upsloping","Flat","Downsloping"].index(slope_display)

        ca = st.selectbox("Major Vessels (0–3)", [0,1,2,3])

        thal_display = st.selectbox("Thal(Maximum heart rate achieved)",
        ["Normal","Fixed Defect","Reversible Defect"])
        thal = [1,2,3][["Normal","Fixed Defect","Reversible Defect"].index(thal_display)]

    if st.button("Predict Heart Risk"):

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        input_scaled = heart_scaler.transform(input_data)

        prob = heart_model.predict_proba(input_scaled)[0][1]

        risk = get_risk(prob)

        if risk == "HIGH":
            st.error(f"Heart Disease Risk: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"Heart Disease Risk: {risk}")
        else:
            st.success(f"Heart Disease Risk: {risk}")

        heart_recommendation(risk)

# ================= DIABETES ====================
if disease == "Diabetes Prediction":

    st.header("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input( "Number of Pregnancies", min_value=0, max_value=20, value=1, help="Enter number of times you have been pregnant")
        glucose = st.number_input( "Glucose Level (mg/dl)", min_value=50, max_value=300, value=100, help="Normal: 70–140 | Prediabetes: 140–199 | Diabetes: >200" )
        bp = st.number_input( "Blood Pressure (mm Hg)", min_value=40, max_value=180, value=80, help="Normal: 80–120" )
        skin = st.number_input( "Skin Thickness (mm)", min_value=5, max_value=100, value=20, help="Normal range: 10–50" )

    with col2:
        insulin = st.number_input( "Insulin Level (µU/ml)", min_value=0, max_value=900, value=80, help="Normal: 16–166" )
        bmi = st.number_input( "Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.0, help="Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: >30" )
        dpf = st.number_input( "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, help="Higher value means higher hereditary risk" )
        age = st.number_input( "Age", min_value=1, max_value=120, value=30, help="Enter your age in years" )

    if st.button("Predict Diabetes Risk"):

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

        input_scaled = diabetes_scaler.transform(input_data)

        prob = diabetes_model.predict_proba(input_scaled)[0][1]

        risk = get_risk(prob)

        if risk == "HIGH":
            st.error(f"Diabetes Risk: {risk}")
        elif risk == "MEDIUM":
            st.warning(f"Diabetes Risk: {risk}")
        else:
            st.success(f"Diabetes Risk: {risk}")

        diabetes_recommendation(risk)
