"""
ML Capstone Prediction Suite - Interactive Web Application
===========================================================
Dual-Track GUI for B.Tech Machine Learning Capstone (Review 2 Bonus):
  Track 1 (Classification): Bank Marketing Term Deposit Subscription Prediction
  Track 2 (Regression): Seoul Bike Sharing Hourly Demand Prediction

Run with:
    streamlit run app/streamlit_app.py
"""

import datetime
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.inference import (
    BIKE_CATEGORICAL_OPTIONS,
    CATEGORICAL_OPTIONS,
    load_metadata,
    load_model,
    load_regression_metadata,
    load_regression_model,
    predict,
    predict_regression,
    prepare_input,
    prepare_regression_input,
)

# ---------------------------------------------------------------------------
# Page configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Prediction Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .result-yes {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-no {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-bike {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1e9fc 100%);
        border-left: 6px solid #1f77b4;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification pipeline...")
def get_classification_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_classification_metadata():
    return load_metadata()


@st.cache_resource(show_spinner="Loading regression pipeline...")
def get_regression_model():
    return load_regression_model()


@st.cache_data(show_spinner=False)
def get_regression_metadata():
    return load_regression_metadata()


# ---------------------------------------------------------------------------
# Sidebar Navigation & Track Selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎯 Capstone Problem Tracks")
    track = st.radio(
        "Select Machine Learning Track:",
        [
            "🏦 Track 1: Bank Marketing (Classification)",
            "🚲 Track 2: Seoul Bike Sharing (Regression)",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 📌 Project Context")
    if "Classification" in track:
        st.markdown(
            """
            **Classification Track**
            - **Task:** Term Deposit Subscription (`y = yes / no`)
            - **Dataset:** UCI Bank Marketing (`bank-full.csv`, 45,211 rows)
            - **Model:** Gradient Boosting Classifier
            - **Metrics:** 91.08% Accuracy, 0.9323 ROC-AUC
            """
        )
    else:
        st.markdown(
            """
            **Regression Track**
            - **Task:** Hourly Rented Bike Count
            - **Dataset:** Seoul Bike Sharing (`SeoulBikeData.csv`, 8,760 rows)
            - **Model:** Gradient Boosting Regressor
            - **Metrics:** 0.9275 $R^2$, 168.70 RMSE
            """
        )

    st.divider()
    st.markdown("#### Team")
    st.markdown("Ritvik &bull; Abhisarg &bull; Rohan", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Academic Disclaimer:</strong> This application demonstrates trained ML models
        for an academic capstone evaluation. Outputs are statistical predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# TRACK 1: BANK MARKETING (CLASSIFICATION)
# ===========================================================================
if "Classification" in track:
    st.markdown('<p class="main-header">🏦 Bank Marketing Term Deposit Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predict whether a bank client will subscribe to a term deposit '
        "based on their demographic, financial profile, and marketing campaign history.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        clf_pipeline = get_classification_model()
        clf_metadata = get_classification_metadata()
        clf_ready = True
    except FileNotFoundError as e:
        st.error(f"**Classification model artifact not found.** {e}\nRun `python scripts/export_model.py`.")
        clf_ready = False
    except Exception as e:
        st.error(f"**Failed to load classification model:** {e}")
        clf_ready = False

    if clf_ready:
        st.markdown("### Enter Client & Campaign Information")

        with st.form("clf_form"):
            st.markdown("#### 👤 Customer Profile")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            with c2:
                job = st.selectbox("Job", options=CATEGORICAL_OPTIONS["job"], index=4)
            with c3:
                marital = st.selectbox("Marital Status", options=CATEGORICAL_OPTIONS["marital"], index=1)
            with c4:
                education = st.selectbox("Education", options=CATEGORICAL_OPTIONS["education"], index=1)

            st.markdown("#### 💰 Financial Information")
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                balance = st.number_input("Average Yearly Balance (EUR)", min_value=-10000, max_value=200000, value=1000, step=100)
            with c6:
                default = st.selectbox("Credit Default?", options=["no", "yes"], index=0)
            with c7:
                housing = st.selectbox("Housing Loan?", options=["no", "yes"], index=0)
            with c8:
                loan = st.selectbox("Personal Loan?", options=["no", "yes"], index=0)

            st.markdown("#### 📞 Campaign Contact Details")
            c9, c10, c11, c12 = st.columns(4)
            with c9:
                contact = st.selectbox("Contact Communication Type", options=CATEGORICAL_OPTIONS["contact"], index=0)
            with c10:
                day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15, step=1)
            with c11:
                month = st.selectbox("Last Contact Month", options=CATEGORICAL_OPTIONS["month"], index=4)
            with c12:
                duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=180, step=10)

            st.markdown("#### 📊 Previous Campaign History")
            c13, c14, c15, c16 = st.columns(4)
            with c13:
                campaign = st.number_input("Contacts During Campaign", min_value=1, max_value=100, value=1, step=1)
            with c14:
                pdays = st.number_input("Days Since Previous Campaign (-1 = none)", min_value=-1, max_value=999, value=-1, step=1)
            with c15:
                previous = st.number_input("Previous Contacts Count", min_value=0, max_value=100, value=0, step=1)
            with c16:
                poutcome = st.selectbox("Previous Campaign Outcome", options=CATEGORICAL_OPTIONS["poutcome"], index=3)

            st.divider()
            b1, b2, b3 = st.columns([1, 2, 1])
            with b2:
                clf_submitted = st.form_submit_button("🔮 Predict Term Deposit Subscription", use_container_width=True, type="primary")

        if clf_submitted:
            raw_input = {
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day": day, "month": month, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
            }
            try:
                input_df = prepare_input(raw_input)
                result = predict(clf_pipeline, input_df)

                st.markdown("---")
                st.markdown("### Prediction Result")

                if result["predicted_class"] == "yes":
                    st.markdown(
                        '<div class="result-yes">'
                        "<h3>✅ Likely to Subscribe</h3>"
                        "<p>The model predicts this client is <strong>likely</strong> to subscribe to a term deposit.</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-no">'
                        "<h3>❌ Unlikely to Subscribe</h3>"
                        "<p>The model predicts this client is <strong>unlikely</strong> to subscribe to a term deposit.</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                if result["probabilities"]:
                    st.markdown("#### Estimated Class Probabilities")
                    p1, p2 = st.columns(2)
                    with p1:
                        st.metric("P(No Subscription)", f"{result['probabilities']['no']:.1%}")
                    with p2:
                        st.metric("P(Subscription)", f"{result['probabilities']['yes']:.1%}")
                    st.progress(result["probabilities"]["yes"])

                st.info(f"**Inference Model:** {result['model_name']}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Transparency expander
        with st.expander("📈 Classification Model Performance & Transparency", expanded=False):
            if clf_metadata:
                metrics = clf_metadata.get("evaluation_metrics", {})
                st.markdown(f"**Model:** {clf_metadata.get('model_name', 'N/A')}")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                with m2:
                    st.metric("Precision (W)", f"{metrics.get('weighted_precision', 0):.2%}")
                with m3:
                    st.metric("Recall (W)", f"{metrics.get('weighted_recall', 0):.2%}")
                with m4:
                    st.metric("F1-Score (W)", f"{metrics.get('weighted_f1', 0):.2%}")
                with m5:
                    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

                st.markdown("#### Tuned Hyperparameters")
                st.json(clf_metadata.get("model_params", {}))
                st.markdown("#### Preprocessing Pipeline")
                st.json(clf_metadata.get("preprocessing", {}))


# ===========================================================================
# TRACK 2: SEOUL BIKE SHARING (REGRESSION)
# ===========================================================================
else:
    st.markdown('<p class="main-header">🚲 Seoul Bike Sharing Demand Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predict hourly rented bike demand in Seoul based on calendar schedule, '
        "ambient weather conditions, and environmental factors.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        reg_pipeline = get_regression_model()
        reg_metadata = get_regression_metadata()
        reg_ready = True
    except FileNotFoundError as e:
        st.error(f"**Regression model artifact not found.** {e}\nRun `python scripts/export_regression_model.py`.")
        reg_ready = False
    except Exception as e:
        st.error(f"**Failed to load regression model:** {e}")
        reg_ready = False

    if reg_ready:
        st.markdown("### Enter Date, Time & Environmental Conditions")

        with st.form("reg_form"):
            # --- Section 1: Temporal & Calendar ---
            st.markdown("#### 📅 Temporal & Calendar Settings")
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                selected_date = st.date_input(
                    "Date",
                    value=datetime.date(2018, 6, 15),
                    help="Determines day of week and weekend status.",
                )
            with r2:
                hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=18, step=1)
            with r3:
                seasons = st.selectbox("Season", options=BIKE_CATEGORICAL_OPTIONS["seasons"], index=1)  # Summer
            with r4:
                holiday = st.selectbox("Holiday Status", options=BIKE_CATEGORICAL_OPTIONS["holiday"], index=0)
            with r5:
                functioning_day = st.selectbox(
                    "Rental Station Status",
                    options=BIKE_CATEGORICAL_OPTIONS["functioning_day"],
                    index=0,
                    help="On non-functioning days, stations are closed and demand is deterministically 0.",
                )

            # --- Section 2: Temperature & Atmospheric ---
            st.markdown("#### ☀️ Temperature & Atmospheric Dynamics")
            r6, r7, r8, r9, r10 = st.columns(5)
            with r6:
                temperature = st.number_input("Temperature (°C)", min_value=-25.0, max_value=45.0, value=25.0, step=0.5)
            with r7:
                humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
            with r8:
                dew_point = st.number_input("Dew Point Temp (°C)", min_value=-35.0, max_value=35.0, value=14.0, step=0.5)
            with r9:
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)
            with r10:
                solar_rad = st.number_input("Solar Radiation (MJ/m²)", min_value=0.0, max_value=5.0, value=1.8, step=0.1)

            # --- Section 3: Weather Precipitation & Visibility ---
            st.markdown("#### 🌧️ Precipitation & Visibility")
            r11, r12, r13 = st.columns(3)
            with r11:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=60.0, value=0.0, step=0.5)
            with r12:
                snowfall = st.number_input("Snowfall (cm)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
            with r13:
                visibility = st.number_input("Visibility (10m units)", min_value=0, max_value=2000, value=1800, step=50)

            st.divider()
            rb1, rb2, rb3 = st.columns([1, 2, 1])
            with rb2:
                reg_submitted = st.form_submit_button("🚲 Predict Hourly Bike Demand", use_container_width=True, type="primary")

        if reg_submitted:
            raw_bike_input = {
                "date": selected_date.strftime("%d/%m/%Y"),
                "hour": hour,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "visibility": visibility,
                "dew_point_temperature": dew_point,
                "solar_radiation": solar_rad,
                "rainfall": rainfall,
                "snowfall": snowfall,
                "seasons": seasons,
                "holiday": holiday,
                "functioning_day": functioning_day,
            }

            try:
                is_functioning = (functioning_day == "Yes")
                input_df = prepare_regression_input(raw_bike_input)
                result = predict_regression(reg_pipeline, input_df, is_functioning=is_functioning)

                st.markdown("---")
                st.markdown("### Prediction Result")

                if not is_functioning:
                    st.warning(
                        "🛑 **Rental Stations Non-Functioning:** Rental stations are recorded as closed "
                        "on maintenance / non-functioning days. Demand is deterministically **0 bikes**."
                    )
                else:
                    count_int = result["predicted_count_int"]
                    # Demand level classification
                    if count_int > 2000:
                        level_badge = "🔥 **Peak Demand** (Commuter Rush / Ideal Weather)"
                    elif count_int > 1000:
                        level_badge = "📈 **High Demand** (Active Transit Hours)"
                    elif count_int > 300:
                        level_badge = "⚖️ **Moderate Demand** (Standard Operations)"
                    else:
                        level_badge = "🌙 **Low Demand** (Off-peak / Inclement Weather)"

                    st.markdown(
                        f"""
                        <div class="result-bike">
                            <h2 style="margin:0; color:#1f4e79;">🚲 {count_int:,} Rented Bikes</h2>
                            <p style="margin:0.4rem 0 0 0; font-size:1.05rem;">{level_badge}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Diagnostic Breakdown Cards
                    dow_name = selected_date.strftime("%A")
                    is_rush = "Yes (Peak Commuter Window)" if (selected_date.weekday() < 5 and hour in [7, 8, 9, 17, 18, 19]) else "No"
                    th_idx = temperature * (humidity / 100.0)

                    c_d1, c_d2, c_d3 = st.columns(3)
                    with c_d1:
                        st.metric("Day of Week", f"{dow_name} ({'Weekend' if selected_date.weekday() >= 5 else 'Weekday'})")
                    with c_d2:
                        st.metric("Rush Hour Window", is_rush)
                    with c_d3:
                        st.metric("Temp-Humidity Index", f"{th_idx:.1f}")

                st.info(f"**Inference Model:** {result['model_name']}")

            except Exception as e:
                st.error(f"Regression prediction failed: {e}")

        # Model Performance & Transparency
        with st.expander("📈 Regression Model Performance & Transparency", expanded=False):
            if reg_metadata:
                metrics = reg_metadata.get("evaluation_metrics", {})
                st.markdown(f"**Selected Model:** {reg_metadata.get('model_name', 'N/A')}")
                st.markdown(
                    "**Model Selection Rationale:** Evaluated in the regression track alongside 10 models "
                    "(Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN). "
                    "Gradient Boosting achieves the highest predictive accuracy with an $R^2$ of **0.9275** and low RMSE on unseen test data."
                )

                rm1, rm2, rm3 = st.columns(3)
                with rm1:
                    st.metric("R² Score (Test Split)", f"{metrics.get('r2_score', 0):.4f}")
                with rm2:
                    st.metric("RMSE (Hourly Count)", f"{metrics.get('rmse', 0):.2f}")
                with rm3:
                    st.metric("MAE (Hourly Count)", f"{metrics.get('mae', 0):.2f}")

                st.markdown("#### Model Hyperparameters")
                st.json(reg_metadata.get("model_params", {}))

                st.markdown("#### Preprocessing & Engineered Features")
                st.json(reg_metadata.get("preprocessing", {}))

                st.markdown(
                    """
                    #### Methodology Notes
                    - **Leakage Prevention:** `StandardScaler` and `OneHotEncoder` fitted exclusively on training set.
                    - **Deterministic Filter:** Filtered non-functioning days (`functioning_day == 'No'`) during training.
                    - **Train-Test Split:** 80/20 split with fixed `random_state=42`.
                    - **Feature Engineering:** `day_of_week`, `is_weekend`, `rush_hour`, and `temp_humidity_idx`.
                    """
                )

        # Sample test scenarios
        with st.expander("🧪 Try Sample Scenarios (Seoul Bike)", expanded=False):
            st.markdown(
                """
                | Scenario | Date | Hour | Season | Temp | Rainfall | Expected Demand |
                |---|---|---|---|---|---|---|
                | **Weekday Evening Rush** | Friday 15/06/2018 | 18:00 | Summer | 26°C | 0.0 mm | **Peak (> 2,500 bikes)** |
                | **Weekday Morning Rush** | Tuesday 08/05/2018 | 08:00 | Spring | 18°C | 0.0 mm | **High (~ 1,500 - 2,000)** |
                | **Heavy Rainy Afternoon** | Wednesday 11/07/2018 | 14:00 | Summer | 22°C | 15.0 mm | **Low (< 200 bikes)** |
                | **Sub-zero Winter Night** | Monday 15/01/2018 | 03:00 | Winter | -8°C | 0.0 mm | **Very Low (< 100 bikes)** |
                | **Maintenance Day** | Any | Any | Any | Any | Any | **0 bikes (Closed)** |
                """
            )
