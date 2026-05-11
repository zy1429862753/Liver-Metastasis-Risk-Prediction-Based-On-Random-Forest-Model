import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 变量映射配置 (全 15 个临床变量，完美适配肝转移字典)
# ==========================================
VAR_CONFIG = {
    "Age": {"<80 years (Code: 0)": 0, "≥80 years (Code: 1)": 1},
    "Sex": {"Female (Code: 0)": 0, "Male (Code: 1)": 1},
    "Race": {"Black (Code: 0)": 0, "White (Code: 1)": 1, "Others (Code: 2)": 2},
    "Grade": {"I/II (Code: 0)": 0, "III/IV (Code: 1)": 1},
    "Histological_type": {"Adenocarcinoma (Code: 0)": 0, "Others (Code: 1)": 1},
    "T_stage": {"T0 (Code: 0)": 0, "T1 (Code: 1)": 1, "T2 (Code: 2)": 2, "T3 (Code: 3)": 3, "T4 (Code: 4)": 4},
    "N_stage": {"N0 (Code: 0)": 0, "N+ (Code: 1)": 1},
    "Bone_metastasis": {"No (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Brain_metastasis": {"No (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Lung_metastasis": {"No (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Tumor_size": {"<2.0 cm (Code: 0)": 0, "≥2.0 cm (Code: 1)": 1},
    "Surgery": {"No (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Radiation": {"None/Unknown (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Chemotherapy": {"No/Unknown (Code: 0)": 0, "Yes (Code: 1)": 1},
    "Marital_status": {"Unmarried (Code: 0)": 0, "Married (Code: 1)": 1}
}

# ==========================================
# 2. 页面配置与 UI 样式
# ==========================================
st.set_page_config(page_title="Liver Metastasis Risk Tool", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important;}
    .main-header {
        text-align: center; color: #333; margin-bottom: 20px; 
        font-weight: 700; font-size: 28px;
    }
    .custom-label {
        font-size: 16px !important; font-weight: 600; 
        color: #444; margin-top: 15px; margin-bottom: 5px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 15px;
    }
    div.stButton > button {
        background-color: #1f77b4; color: white; font-size: 18px; 
        height: 3em; border-radius: 8px; width: 100%; font-weight: bold;
    }
    .explanation-title {
        color: #444; font-size: 20px; font-weight: 600; margin-top: 20px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 加载模型
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("rf_model_deploy.pkl")
        features = joblib.load("feature_names_rf.pkl")
        return model, features
    except Exception as e:
        st.error(f"Failed to load model or features: {e}")
        return None, []


model, feature_names = load_model()

# ==========================================
# 4. 界面逻辑
# ==========================================
st.markdown("<div class='main-header'>Liver Metastasis Risk Prediction Based On Random Forest</div>",
            unsafe_allow_html=True)

user_input_values = {}

if not model:
    st.error("⚠️ 未找到模型或特征文件，请确认 rf_model_deploy.pkl 和 feature_names_rf.pkl 在同一目录下。")
else:
    col_input, col_result = st.columns([2, 2], gap="large")

    with col_input:
        with st.container(border=True):
            st.markdown("### Patient Parameters")
            cols = st.columns(2)

            for idx, feature in enumerate(feature_names):
                current_col = cols[idx % 2]
                with current_col:
                    display_name = feature.replace('_', ' ')
                    st.markdown(f"<div class='custom-label'>{display_name}</div>", unsafe_allow_html=True)

                    if feature in VAR_CONFIG:
                        options_map = VAR_CONFIG[feature]
                        options_labels = list(options_map.keys())

                        selected_label = st.radio(
                            label=f"radio_{feature}",
                            options=options_labels,
                            key=feature,
                            label_visibility="collapsed",
                            horizontal=True
                        )
                        user_input_values[feature] = options_map[selected_label]

    with col_result:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("### Prediction Result & Explanation")

            res_ph = st.empty()
            chart_ph = st.empty()
            shap_ph = st.empty()

            if st.button("🚀 Calculate Risk"):
                try:
                    input_df = pd.DataFrame([user_input_values], columns=feature_names)

                    # 获取经过校准后的真实临床概率
                    pred_prob = model.predict_proba(input_df)[0][1]
                    risk_percent = pred_prob * 100

                    # 💡 核心修改：基于真实 25% 阳性率设定的全新阈值
                    if risk_percent < 20:
                        bar_color = "#2ca02c"  # 绿色
                        res_ph.success(f"**Low Risk**: The probability of Liver Metastasis is {risk_percent:.1f}%")
                    elif risk_percent < 40:
                        bar_color = "#ff7f0e"  # 橙色
                        res_ph.warning(f"**Medium Risk**: The probability of Liver Metastasis is {risk_percent:.1f}%")
                    else:
                        bar_color = "#d62728"  # 红色
                        res_ph.error(f"**High Risk**: The probability of Liver Metastasis is {risk_percent:.1f}%")

                    # 绘制仪表盘
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_percent,
                        number={'suffix': "%", 'font': {'size': 35, 'color': "#333"}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': bar_color},
                            'bgcolor': "white",
                            'steps': [{'range': [0, 100], 'color': '#f0f2f6'}],
                            'threshold': {
                                'line': {'color': "black", 'width': 3},
                                'thickness': 0.75,
                                'value': risk_percent
                            }
                        }
                    ))
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
                    chart_ph.plotly_chart(fig, use_container_width=True)

                    # --- SHAP 瀑布图 ---
                    st.markdown("<div class='explanation-title'>Why this prediction? (SHAP Waterfall Plot)</div>",
                                unsafe_allow_html=True)
                    # 💡 修改图注，向临床医生解释 SHAP 分数与上方真实概率的关系
                    st.caption(
                        "Note: The SHAP plot illustrates the model's internal risk scoring logic. Red bars increase the internal score, blue bars decrease it. This internal score is then mathematically calibrated to the true probability shown above.")

                    with st.spinner('Generating Explanation...'):

                        # ==============================================================
                        # 💡 核心修复：从校准器中提取纯正的随机森林模型，供 SHAP 解析
                        # 兼容不同版本的 scikit-learn (1.2+ 使用 estimator，旧版使用 base_estimator)
                        # ==============================================================
                        if hasattr(model, 'calibrated_classifiers_'):
                            calibrated_clf = model.calibrated_classifiers_[0]
                            if hasattr(calibrated_clf, 'estimator'):
                                shap_model = calibrated_clf.estimator
                            else:
                                shap_model = calibrated_clf.base_estimator
                        else:
                            shap_model = model

                        # 使用提取出来的基础随机森林模型
                        explainer = shap.TreeExplainer(shap_model)
                        shap_explanation = explainer(input_df)

                        if len(shap_explanation.shape) == 3:
                            explanation_pos = shap_explanation[0, :, 1]
                        else:
                            explanation_pos = shap_explanation[0]

                        fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
                        explanation_pos.feature_names = [name.replace('_', ' ') for name in feature_names]

                        shap.plots.waterfall(explanation_pos, show=False)
                        st.pyplot(fig_shap, bbox_inches='tight')
                        plt.close(fig_shap)

                except Exception as e:
                    st.error(f"Prediction Error: {str(e)}")
            else:
                chart_ph.info("Click 'Calculate Risk' to see the probability and explanation.")