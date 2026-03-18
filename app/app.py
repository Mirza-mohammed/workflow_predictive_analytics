import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "models/final_gradient_boosting_pipeline.joblib"


def duration_band(hours: float) -> str:
    if hours < 24:
        return "Short"
    elif hours <= 72:
        return "Medium"
    return "Long"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Workflow Duration Predictor", layout="centered")
    st.title("Workflow Ticket Duration Predictor")
    st.write(
        "This prototype predicts ticket resolution time using the final "
        "Gradient Boosting model from the project."
    )

    artifact = load_artifact()
    model = artifact["model"]

    st.subheader("Enter ticket details")

    priority = st.text_input("Priority", value="3 - Moderate / Low")
    category = st.text_input("Category", value="inquiry / help")
    subcategory = st.text_input("Subcategory", value="internal application")
    impact = st.text_input("Impact", value="2 - medium")
    urgency = st.text_input("Urgency", value="2 - medium")
    contact_type = st.text_input("Contact Type", value="phone")
    location = st.text_input("Location", value="location 1")
    u_symptom = st.text_input("Symptom", value="symptom 1")
    cmdb_ci = st.text_input("Configuration Item", value="ci 1")

    opened_hour = st.slider("Opened Hour", min_value=0, max_value=23, value=9)
    opened_dayofweek = st.slider("Opened Day of Week", min_value=0, max_value=6, value=1)

    is_weekend = 1 if opened_dayofweek in [5, 6] else 0
    high_priority_flag = 1 if str(priority).strip().lower() in [
        "1 - critical", "2 - high", "high", "critical"
    ] else 0

    if st.button("Predict Resolution Time"):
        input_df = pd.DataFrame([{
            "priority": priority,
            "category": category,
            "subcategory": subcategory,
            "impact": impact,
            "urgency": urgency,
            "contact_type": contact_type,
            "location": location,
            "u_symptom": u_symptom,
            "cmdb_ci": cmdb_ci,
            "opened_hour": opened_hour,
            "opened_dayofweek": opened_dayofweek,
            "is_weekend": is_weekend,
            "high_priority_flag": high_priority_flag
        }])

        pred_log = model.predict(input_df)[0]
        pred_hours = float(np.expm1(pred_log))
        pred_hours = max(pred_hours, 0.0)

        st.success(f"Predicted resolution time: {pred_hours:.2f} hours")
        st.info(f"Predicted duration band: {duration_band(pred_hours)}")


if __name__ == "__main__":
    main()