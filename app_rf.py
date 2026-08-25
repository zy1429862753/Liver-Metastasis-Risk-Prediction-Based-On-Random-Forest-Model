from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "rf_model_deploy.pkl"
FEATURE_PATH = APP_DIR / "feature_names_rf.pkl"
CONFIG_PATH = APP_DIR / "rf_deployment_config.json"


VAR_CONFIG = {
    "Age": {"<80 years": 0, "≥80 years": 1},
    "Grade": {"Grade I/II": 0, "Grade III/IV": 1},
    "Histological_type": {"Adenocarcinoma": 0, "Other histological type": 1},
    "T_stage": {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4},
    "N_stage": {"N0": 0, "N+": 1},
    "Tumor_size": {"<2.0 cm": 0, "≥2.0 cm": 1},
    "Marital_status": {"Unmarried": 0, "Married": 1},
}

DISPLAY_NAMES = {
    "Age": "Age",
    "Grade": "Histological grade",
    "Histological_type": "Histological type",
    "T_stage": "T stage",
    "N_stage": "N stage",
    "Tumor_size": "Tumor size",
    "Marital_status": "Marital status",
}


st.set_page_config(
    page_title="Liver Metastasis Risk Tool",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem !important; max-width: 1250px;}
        .main-header {
            text-align: center; color: #333; margin-bottom: 8px;
            font-weight: 700; font-size: 28px;
        }
        .sub-header {
            text-align: center; color: #666; margin-bottom: 22px; font-size: 15px;
        }
        .custom-label {
            font-size: 16px !important; font-weight: 600;
            color: #444; margin-top: 15px; margin-bottom: 5px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f8f9fa; border: 1px solid #ddd;
            border-radius: 8px; padding: 15px;
        }
        div.stButton > button {
            background-color: #1f77b4; color: white; font-size: 18px;
            height: 3em; border-radius: 8px; width: 100%; font-weight: bold;
        }
        .explanation-title {
            color: #444; font-size: 20px; font-weight: 600;
            margin-top: 20px; margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_deployment_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if features != config.get("features"):
        raise ValueError("Feature order differs between feature_names_rf.pkl and config.")
    missing_ui_config = [feature for feature in features if feature not in VAR_CONFIG]
    if missing_ui_config:
        raise ValueError(f"Missing UI mapping for: {missing_ui_config}")
    if not 0 < float(config["decision_threshold"]) < 1:
        raise ValueError("The deployment decision threshold must be between 0 and 1.")
    return model, features, config


def positive_class_explanation(model, input_frame: pd.DataFrame, feature_names: list[str]):
    explainer = shap.TreeExplainer(model)
    explanation = explainer(input_frame)

    if explanation.values.ndim == 3:
        positive = explanation[0, :, 1]
    elif explanation.values.ndim == 2:
        positive = explanation[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {explanation.values.shape}")
    positive.feature_names = [DISPLAY_NAMES[name] for name in feature_names]
    return positive


st.markdown(
    "<div class='main-header'>Liver Metastasis Risk Prediction Based on Random Forest</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-header'>Seven-variable research model with an independently locked decision threshold</div>",
    unsafe_allow_html=True,
)

try:
    model, feature_names, deployment_config = load_deployment_artifacts()
except Exception as error:
    model, feature_names, deployment_config = None, [], {}
    st.error(
        "Failed to load the deployment files. Confirm that rf_model_deploy.pkl, "
        "feature_names_rf.pkl, and rf_deployment_config.json are in the App directory. "
        f"Details: {error}"
    )


if model is not None:
    decision_threshold = float(deployment_config["decision_threshold"])
    threshold_percent = decision_threshold * 100
    user_input_values: dict[str, int] = {}

    col_input, col_result = st.columns([2, 2], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown("### Patient parameters")
            cols = st.columns(2)

            for index, feature in enumerate(feature_names):
                with cols[index % 2]:
                    st.markdown(
                        f"<div class='custom-label'>{DISPLAY_NAMES[feature]}</div>",
                        unsafe_allow_html=True,
                    )
                    options_map = VAR_CONFIG[feature]
                    selected_label = st.radio(
                        label=f"radio_{feature}",
                        options=list(options_map),
                        key=feature,
                        label_visibility="collapsed",
                        horizontal=True,
                    )
                    user_input_values[feature] = options_map[selected_label]

    with col_result:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Prediction result and explanation")
            result_placeholder = st.empty()
            chart_placeholder = st.empty()

            if st.button("Calculate risk", type="primary"):
                try:
                    input_frame = pd.DataFrame(
                        [user_input_values], columns=feature_names, dtype=float
                    )
                    predicted_probability = float(model.predict_proba(input_frame)[0, 1])
                    risk_percent = predicted_probability * 100
                    above_threshold = predicted_probability >= decision_threshold

                    if above_threshold:
                        bar_color = "#d62728"
                        result_placeholder.error(
                            f"**Higher model-predicted risk:** {risk_percent:.1f}% "
                            f"(at or above the locked {threshold_percent:.1f}% threshold)"
                        )
                    else:
                        bar_color = "#2ca02c"
                        result_placeholder.success(
                            f"**Lower model-predicted risk:** {risk_percent:.1f}% "
                            f"(below the locked {threshold_percent:.1f}% threshold)"
                        )

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_percent,
                            number={"suffix": "%", "font": {"size": 35, "color": "#333"}},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": bar_color},
                                "bgcolor": "white",
                                "steps": [{"range": [0, 100], "color": "#f0f2f6"}],
                                "threshold": {
                                    "line": {"color": "black", "width": 3},
                                    "thickness": 0.75,
                                    "value": threshold_percent,
                                },
                            },
                        )
                    )
                    gauge.update_layout(height=220, margin=dict(l=20, r=20, t=35, b=10))
                    chart_placeholder.plotly_chart(gauge, width="stretch")
                    st.caption(
                        f"Black line: {threshold_percent:.1f}% cutoff selected from five-fold "
                        "out-of-fold predictions in the training cohort using the Youden index."
                    )

                    st.markdown(
                        "<div class='explanation-title'>Why this prediction? (SHAP waterfall plot)</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Red features increase and blue features decrease the Random Forest "
                        "probability estimate for this patient."
                    )
                    with st.spinner("Generating explanation..."):
                        positive_explanation = positive_class_explanation(
                            model, input_frame, feature_names
                        )
                        plt.figure(figsize=(8.5, 5.0))
                        shap.plots.waterfall(
                            positive_explanation,
                            max_display=len(feature_names),
                            show=False,
                        )
                        shap_figure = plt.gcf()
                        shap_figure.tight_layout()
                        st.pyplot(shap_figure, width="stretch")
                        plt.close(shap_figure)

                except Exception as error:
                    st.error(f"Prediction error: {error}")
            else:
                chart_placeholder.info("Click 'Calculate risk' to view the result and explanation.")

    st.divider()
    st.caption(
        "Research use only. The displayed value is an uncalibrated Random Forest probability "
        "estimate and must not be interpreted as a diagnosis or replace clinical judgement."
    )
