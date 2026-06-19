import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

MODEL_PATH = Path(__file__).parent.parent / "models" / "bank_marketing_model.joblib"

CATEGORICAL_OPTIONS = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ],
    "marital": ["married", "divorced", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ],
    "poutcome": ["failure", "nonexistent", "success"]
}

NUMERIC_DEFAULTS = {
    "age": (35, 18, 100, 1),
    "balance": (1200, -5000, 20000, 50),
    "day": (15, 1, 31, 1),
    "duration": (180, 1, 2000, 10),
    "campaign": (2, 1, 20, 1),
    "pdays": (-1, -1, 1000, 1),
    "previous": (0, 0, 20, 1)
}


@st.cache_resource
def load_model() -> object:
    return joblib.load(MODEL_PATH)


def build_input_data(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])


def initialize_history():
    if "predictions" not in st.session_state:
        st.session_state.predictions = []


def render_sidebar() -> dict:
    st.sidebar.header("Client profile")
    st.sidebar.write("Enter the client attributes to generate a subscription prediction.")

    with st.sidebar.form(key="client_form"):
        values = {}
        for field, (default, min_value, max_value, step) in NUMERIC_DEFAULTS.items():
            values[field] = st.number_input(
                label=field.replace("_", " ").title(),
                value=default,
                min_value=min_value,
                max_value=max_value,
                step=step,
                key=f"num_{field}"
            )

        for field, options in CATEGORICAL_OPTIONS.items():
            values[field] = st.selectbox(
                label=field.replace("_", " ").title(),
                options=options,
                index=0,
                key=f"cat_{field}"
            )

        submit_button = st.form_submit_button(label="Submit prediction")

    return values if submit_button else {}


def render_result(prediction_label: str, prediction_probability: float, input_values: dict):
    st.success("Prediction completed successfully")

    percent_prob = prediction_probability * 100
    st.markdown("### Prediction overview")
    col1, col2, col3 = st.columns([1.4, 1, 0.8])
    col1.metric("Subscription prediction", prediction_label.upper())
    col2.metric("Subscription probability", f"{percent_prob:.1f}%")
    col3.metric("Model", "Random Forest")

    st.progress(prediction_probability)

    with st.expander("View input values"):
        st.table(pd.DataFrame([input_values]).T.rename(columns={0: "Value"}))

    with st.expander("Prediction interpretation"):
        if prediction_label == "yes":
            st.write(
                "This client is predicted to subscribe to the term deposit. Review the probability and consider the client’s financial profile when planning follow-up actions."
            )
        else:
            st.write(
                "This client is predicted not to subscribe. You may want to adjust outreach strategies or prioritize higher-probability leads."
            )


def add_history(prediction_label: str, prediction_probability: float, input_values: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "prediction": prediction_label,
        "probability": round(prediction_probability, 4),
        **input_values
    }
    st.session_state.predictions.insert(0, entry)


def render_history():
    if st.session_state.predictions:
        st.markdown("---")
        st.markdown("## Prediction history")
        st.write(
            "Review past submissions in this session. Use this table to compare recent client profiles and model output."
        )
        st.dataframe(pd.DataFrame(st.session_state.predictions), width=True)


def main():
    st.set_page_config(
        page_title="Bank Marketing Prediction",
        page_icon="💼",
        layout="wide"
    )

    st.title("Bank Marketing Campaign Prediction")
    st.write(
        "Use a professional interface to submit a client profile and view the subscription prediction. "
        "The model estimates if the client is likely to subscribe to the term deposit product."
    )

    initialize_history()
    client_inputs = render_sidebar()

    if client_inputs:
        try:
            model = load_model()
            input_df = build_input_data(client_inputs)
            pred_class = int(model.predict(input_df)[0])
            pred_prob = float(model.predict_proba(input_df)[:, 1][0])
            label = "yes" if pred_class == 1 else "no"
            render_result(label, pred_prob, client_inputs)
            add_history(label, pred_prob, client_inputs)
        except Exception as error:
            st.error(f"Unable to generate prediction: {error}")

    if st.session_state.predictions:
        render_history()

    with st.expander("About this dashboard"):
        st.write(
            "This Streamlit app uses the same trained bank marketing model as the FastAPI service. "
            "Enter client attributes on the left, submit the form, and review the model output instantly."
        )
        st.write("- Model artifact: bank_marketing_model.joblib")
        st.write("- Prediction target: client subscribes to term deposit")
        st.write("- Data source: UCI Bank Marketing dataset")


if __name__ == "__main__":
    main()
