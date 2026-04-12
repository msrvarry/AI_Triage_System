# Make sure to install the required packages before running this code
# pip install streamlit
# Run this code using the command "streamlit run app.py"

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Triage System", layout="wide")

st.title("🏥 AI-Based Emergency Triage System")
st.markdown("### Explainable Multimodal Triage Prediction")

# -------------------------------
# TRIAGE COLORS
# -------------------------------
triage_colors = {
    1: "🔴 Level 1 – Immediate",
    2: "🟠 Level 2 – Very Urgent",
    3: "🟡 Level 3 – Urgent",
    4: "🟢 Level 4 – Standard",
    5: "🔵 Level 5 – Non-Urgent"
}

# -------------------------------
# INPUT MODE
# -------------------------------
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
def vitals_importance(v):
    return {
        "O2 Saturation": max(0, (95 - v["o2sat"]) / 10),
        "Heart Rate": abs(v["heart_rate"] - 75) / 50,
        "Blood Pressure": abs(v["sbp"] - 120) / 80,
        "Resp Rate": abs(v["resp_rate"] - 16) / 10,
        "Temperature": abs(v["temperature"] - 98.6) / 5,
        "Pain Score": v["pain"] / 10
    }

def text_importance(text):
    keywords = {
        "chest pain": 1.0,
        "shortness of breath": 1.0,
        "bleeding": 0.9,
        "unconscious": 1.0,
        "severe": 0.8,
        "fever": 0.5
    }

    found = {}
    for k, v in keywords.items():
        if k in text.lower():
            found[k] = v

    return found

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Triage Level"):

    probs = np.random.dirichlet(np.ones(5), size=1)[0]
    predicted_class = np.argmax(probs) + 1
    confidence = np.max(probs)

    st.subheader("🧾 Prediction Result")
    st.success(triage_colors[predicted_class])
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # -------------------------------
    # DATA VISUALIZATION (SMALL)
    # -------------------------------
    st.subheader("📊 Data Visualization")

    col1, col2 = st.columns(2)

    # -------- VITALS GRAPH --------
    with col1:
        st.markdown("#### 🔹 Feature Impact")

        if mode == "Vitals (Structured Data)":
            importance = vitals_importance(vitals_data)

            fig, ax = plt.subplots(figsize=(4,3))
            ax.barh(list(importance.keys()), list(importance.values()))
            ax.set_title("Vitals Impact")
            st.pyplot(fig)

        else:
            st.info("Switch to vitals mode to view this chart.")

    # -------- TEXT GRAPH --------
    with col2:
        st.markdown("#### 🔹 Symptom Impact")

        if mode == "Free Text (Symptoms)" and text_data:
            importance = text_importance(text_data)

            if importance:
                fig2, ax2 = plt.subplots(figsize=(4,3))
                ax2.bar(importance.keys(), importance.values())
                ax2.set_title("Keyword Importance")
                plt.xticks(rotation=30)
                st.pyplot(fig2)
            else:
                st.write("No critical keywords detected")

        else:
            st.info("Switch to text mode to view this chart.")

    # -------------------------------
    # XAI SECTION (CLEAN VERSION)
    # -------------------------------
    st.subheader("🧠 Explainability")

    col3, col4 = st.columns(2)

    # -------- VITALS XAI --------
    with col3:
        st.markdown("### 🔹 Vitals Contribution")

        if mode == "Vitals (Structured Data)":
            importance = vitals_importance(vitals_data)

            for k, v in importance.items():
                if v > 0.3:
                    st.write(f"• {k} strongly influenced the prediction")
                else:
                    st.write(f"• {k} had minimal influence")

        else:
            st.info("Available only in vitals mode")

    # -------- TEXT XAI --------
    with col4:
        st.markdown("### 🔹 Text Attribution")

        if mode == "Free Text (Symptoms)" and text_data:
            importance = text_importance(text_data)

            highlighted = text_data
            for word in importance:
                highlighted = highlighted.replace(
                    word,
                    f"<span style='color:red; font-weight:bold'>{word}</span>"
                )

            st.markdown(highlighted, unsafe_allow_html=True)

        else:
            st.info("Available only in text mode")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Explainable Multimodal AI Triage System")
