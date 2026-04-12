import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

# -------------------------------
# LOAD EVERYTHING (CACHED)
# -------------------------------
@st.cache_resource
def load_all():
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    text_model = joblib.load("text_logistic_model.pkl")

    df = pd.read_csv("../data/processed_real_mimic_ed.csv")
    df = df.dropna(subset=["chiefcomplaint"])

    X_structured = df.drop(columns=["acuity", "chiefcomplaint"])
    medians = X_structured.median()

    # SHAP explainer
    background = scaler.transform(X_structured.iloc[:50])
    explainer = shap.Explainer(xgb_model.predict_proba, background)

    return xgb_model, scaler, vectorizer, text_model, medians, explainer, X_structured.columns


xgb_model, scaler, vectorizer, text_model, medians, explainer, feature_cols = load_all()

# -------------------------------
# FEATURE NAME MAPPING (FIXED)
# -------------------------------
feature_mapping = {
    "age": "age",
    "heart_rate": "heartrate",
    "sbp": "sbp",
    "resp_rate": "resprate",
    "o2sat": "o2sat",
    "temperature": "temp",
    "pain": "pain"
}

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Triage System", layout="wide")

st.title("🏥 AI-Based Emergency Triage System")
st.markdown("### Explainable Multimodal Triage Prediction")

triage_colors = {
    1: "🔴 Level 1 – Immediate",
    2: "🟠 Level 2 – Very Urgent",
    3: "🟡 Level 3 – Urgent",
    4: "🟢 Level 4 – Standard",
    5: "🔵 Level 5 – Non-Urgent"
}

mode = st.selectbox(
    "Select Input Type",
    ["Vitals (Structured Data)", "Free Text (Symptoms)"]
)

st.markdown("---")

vitals_data = {}
text_data = ""

# -------------------------------
# INPUT UI
# -------------------------------
if mode == "Vitals (Structured Data)":
    st.subheader("📊 Enter Patient Vitals")

    col1, col2, col3 = st.columns(3)

    with col1:
        vitals_data["age"] = st.number_input("Age", 0, 120, 30)
        vitals_data["heart_rate"] = st.number_input("Heart Rate", value=80)
        vitals_data["sbp"] = st.number_input("Systolic BP", value=120)

    with col2:
        vitals_data["resp_rate"] = st.number_input("Respiratory Rate", value=18)
        vitals_data["o2sat"] = st.number_input("Oxygen Saturation (%)", value=98)
        vitals_data["temperature"] = st.number_input("Temperature (°F)", value=98.6)

    with col3:
        vitals_data["pain"] = st.number_input("Pain Score (0-10)", 0, 10, 3)

elif mode == "Free Text (Symptoms)":
    st.subheader("📝 Enter Clinical Description")
    text_data = st.text_area("Chief Complaint", height=120)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def prepare_full_input(vitals_dict):
    full_input = medians.copy()

    for ui_name, model_name in feature_mapping.items():
        if ui_name in vitals_dict and model_name in full_input:
            full_input[model_name] = vitals_dict[ui_name]

    return pd.DataFrame([full_input])


def get_vitals_prediction_and_shap(vitals_dict):
    df_input = prepare_full_input(vitals_dict)
    scaled = scaler.transform(df_input)

    probs = xgb_model.predict_proba(scaled)
    pred_class = np.argmax(probs) + 1
    confidence = np.max(probs)

    shap_values = explainer(scaled)

    shap_vals = shap_values.values[0][:, pred_class - 1]

    contributions = []

    for ui_name, model_name in feature_mapping.items():
        if model_name in feature_cols:
            idx = list(feature_cols).index(model_name)
            contributions.append((ui_name, shap_vals[idx]))

    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    return pred_class, confidence, contributions


def get_text_prediction_and_explanation(text):
    X_input = vectorizer.transform([text])

    probs = text_model.predict_proba(X_input)
    pred_class = np.argmax(probs) + 1
    confidence = np.max(probs)

    feature_names = vectorizer.get_feature_names_out()
    coef = text_model.coef_[pred_class - 1]

    input_array = X_input.toarray()[0]
    contributions = input_array * coef

    nonzero_indices = input_array.nonzero()[0]

    word_contributions = [(feature_names[i], contributions[i]) for i in nonzero_indices]
    word_contributions = sorted(word_contributions, key=lambda x: abs(x[1]), reverse=True)

    top_words = word_contributions[:3]

    return pred_class, confidence, top_words

# -------------------------------
# BUTTON ACTION
# -------------------------------
if st.button("🔍 Predict Triage Level"):

    st.subheader("🧾 Prediction Result")

    if mode == "Vitals (Structured Data)":
        pred_class, confidence, contributions = get_vitals_prediction_and_shap(vitals_data)

        st.success(triage_colors[pred_class])
        #st.info(f"Confidence: {confidence * 100:.2f}%")

        st.subheader("📊 Feature Impact")

        features = [c[0] for c in contributions]
        values = [abs(c[1]) for c in contributions]

        fig, ax = plt.subplots(figsize=(3,2))
        ax.barh(features, values)
        ax.set_title("Vitals Contribution")
        st.pyplot(fig)

    else:
        pred_class, confidence, top_words = get_text_prediction_and_explanation(text_data)

        st.success(triage_colors[pred_class])
        #st.info(f"Confidence: {confidence * 100:.2f}%")

        st.subheader("🧠 Top Influential Words")

        words = [w[0] for w in top_words]
        scores = [abs(w[1]) for w in top_words]

        fig, ax = plt.subplots(figsize=(3,2))
        ax.bar(words, scores)
        ax.set_title("Word Contribution")
        plt.xticks(rotation=30)
        st.pyplot(fig)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Explainable Multimodal AI Triage System")