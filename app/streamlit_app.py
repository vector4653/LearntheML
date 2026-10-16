"""
ML Capstone Prediction & Clustering Suite - Interactive Web Application
========================================================================
Tri-Track GUI for B.Tech Machine Learning Capstone (Review 2 Bonus):
  Track 1 (Classification): Bank Marketing Term Deposit Subscription Prediction
  Track 2 (Regression): Seoul Bike Sharing Hourly Demand Prediction
  Track 3 (Clustering): Spotify Tracks Audio Profile Persona Discovery

Run with:
    streamlit run app/streamlit_app.py
"""

import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup & Module Cache Invalidation (ensures clean hot-reload on Cloud)
# ---------------------------------------------------------------------------
import importlib

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent

for _p in [str(_REPO_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Clear any stale in-memory cached inference module across hot-reloads
for _mod_name in ["app.inference", "inference"]:
    if _mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[_mod_name])
        except Exception:
            sys.modules.pop(_mod_name, None)

try:
    import app.inference as inference
except (ImportError, ModuleNotFoundError):
    import inference

# Classification Track
CATEGORICAL_OPTIONS = getattr(inference, "CATEGORICAL_OPTIONS", {})
load_model = getattr(inference, "load_model")
load_metadata = getattr(inference, "load_metadata")
prepare_input = getattr(inference, "prepare_input")
predict = getattr(inference, "predict")

# Regression Track
BIKE_CATEGORICAL_OPTIONS = getattr(inference, "BIKE_CATEGORICAL_OPTIONS", {})
load_regression_model = getattr(inference, "load_regression_model")
load_regression_metadata = getattr(inference, "load_regression_metadata")
prepare_regression_input = getattr(inference, "prepare_regression_input")
predict_regression = getattr(inference, "predict_regression")

# Clustering Track
SPOTIFY_AUDIO_FEATURES = getattr(
    inference,
    "SPOTIFY_AUDIO_FEATURES",
    [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    ],
)
load_clustering_model = getattr(inference, "load_clustering_model")
load_clustering_metadata = getattr(inference, "load_clustering_metadata")
prepare_clustering_input = getattr(inference, "prepare_clustering_input")
predict_cluster = getattr(inference, "predict_cluster")

# ---------------------------------------------------------------------------
# Page configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Tri-Track Suite",
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
    .result-cluster {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 6px solid #8e24aa;
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


@st.cache_resource(show_spinner="Loading clustering bundle...")
def get_clustering_model():
    return load_clustering_model()


@st.cache_data(show_spinner=False)
def get_clustering_metadata():
    return load_clustering_metadata()


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
            "🎵 Track 3: Spotify Audio Archetypes (Clustering)",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 📌 Project Track Summary")
    if "Classification" in track:
        st.markdown(
            """
            **Classification Track**
            - **Task:** Term Deposit Subscription (`yes / no`)
            - **Dataset:** UCI Bank Marketing (`bank-full.csv`, 45,211 rows)
            - **Model:** Gradient Boosting Classifier
            - **Metrics:** 91.08% Accuracy, 0.9323 ROC-AUC
            """
        )
    elif "Regression" in track:
        st.markdown(
            """
            **Regression Track**
            - **Task:** Hourly Rented Bike Count
            - **Dataset:** Seoul Bike Sharing (`SeoulBikeData.csv`, 8,760 rows)
            - **Model:** Gradient Boosting Regressor
            - **Metrics:** 0.9275 $R^2$, 168.70 RMSE
            """
        )
    else:
        st.markdown(
            """
            **Clustering Track**
            - **Task:** Unsupervised Audio Archetype Discovery
            - **Dataset:** Spotify Tracks (`spotify-tracks-dataset.csv`, 114k rows)
            - **Models:** K-Means ($k=5$) & Hierarchical (Ward)
            - **Metrics:** Silhouette 0.1740, Davies-Bouldin 1.6244
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


# ===========================================================================
# TRACK 2: SEOUL BIKE SHARING (REGRESSION)
# ===========================================================================
elif "Regression" in track:
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
            st.markdown("#### 📅 Temporal & Calendar Settings")
            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                selected_date = st.date_input("Date", value=datetime.date(2018, 6, 15))
            with r2:
                hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=18, step=1)
            with r3:
                seasons = st.selectbox("Season", options=BIKE_CATEGORICAL_OPTIONS["seasons"], index=1)
            with r4:
                holiday = st.selectbox("Holiday Status", options=BIKE_CATEGORICAL_OPTIONS["holiday"], index=0)
            with r5:
                functioning_day = st.selectbox(
                    "Rental Station Status",
                    options=BIKE_CATEGORICAL_OPTIONS["functioning_day"],
                    index=0,
                    help="On non-functioning days, stations are closed and demand is deterministically 0.",
                )

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
                    st.warning("🛑 **Rental Stations Non-Functioning:** Demand is deterministically **0 bikes** on maintenance days.")
                else:
                    count_int = result["predicted_count_int"]
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

                st.info(f"**Inference Model:** {result['model_name']}")
            except Exception as e:
                st.error(f"Regression prediction failed: {e}")

        with st.expander("📈 Regression Model Performance & Transparency", expanded=False):
            if reg_metadata:
                metrics = reg_metadata.get("evaluation_metrics", {})
                st.markdown(f"**Selected Model:** {reg_metadata.get('model_name', 'N/A')}")
                rm1, rm2, rm3 = st.columns(3)
                with rm1:
                    st.metric("R² Score (Test Split)", f"{metrics.get('r2_score', 0):.4f}")
                with rm2:
                    st.metric("RMSE (Hourly Count)", f"{metrics.get('rmse', 0):.2f}")
                with rm3:
                    st.metric("MAE (Hourly Count)", f"{metrics.get('mae', 0):.2f}")


# ===========================================================================
# TRACK 3: SPOTIFY AUDIO ARCHETYPES (CLUSTERING)
# ===========================================================================
else:
    st.markdown('<p class="main-header">🎵 Spotify Audio Profile Clustering</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Discover natural musical archetypes, mood profiles, and song recommendations '
        "using unsupervised K-Means ($k=5$) & Hierarchical Clustering on acoustic descriptors.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        cluster_bundle = get_clustering_model()
        cluster_metadata = get_clustering_metadata()
        cluster_ready = True
    except FileNotFoundError as e:
        st.error(f"**Clustering model artifact not found.** {e}\nRun `python scripts/export_clustering_model.py`.")
        cluster_ready = False
    except Exception as e:
        st.error(f"**Failed to load clustering model:** {e}")
        cluster_ready = False

    if cluster_ready:
        st.markdown("### 🎛️ Tune Audio Descriptors or Choose a Preset")

        preset = st.selectbox(
            "Select an Audio Profile Preset (or customize sliders below):",
            [
                "Custom Audio Profile",
                "Preset: Calm Ambient & Sleep (Low Energy, Acoustic)",
                "Preset: Acoustic Ballad & Romance (Mellow Vocals)",
                "Preset: High-Octane Metal & Rock (Heavy & Loud)",
                "Preset: Live Stage, Standup & Spoken Word",
                "Preset: Dance Pop Fiesta (High Danceability & Valence)",
            ],
            index=0,
        )

        # Preset values dictionary
        preset_values = {
            "Custom Audio Profile": (0.65, 0.70, -6.5, 0.08, 0.15, 0.02, 0.15, 0.65, 120.0),
            "Preset: Calm Ambient & Sleep (Low Energy, Acoustic)": (0.25, 0.12, -22.0, 0.04, 0.92, 0.85, 0.10, 0.15, 75.0),
            "Preset: Acoustic Ballad & Romance (Mellow Vocals)": (0.50, 0.35, -12.0, 0.05, 0.75, 0.02, 0.12, 0.40, 95.0),
            "Preset: High-Octane Metal & Rock (Heavy & Loud)": (0.40, 0.95, -3.5, 0.12, 0.01, 0.35, 0.25, 0.35, 150.0),
            "Preset: Live Stage, Standup & Spoken Word": (0.55, 0.65, -9.0, 0.85, 0.30, 0.00, 0.75, 0.50, 115.0),
            "Preset: Dance Pop Fiesta (High Danceability & Valence)": (0.85, 0.85, -4.5, 0.06, 0.08, 0.00, 0.15, 0.88, 125.0),
        }

        d_dance, d_energy, d_loud, d_speech, d_acoust, d_instru, d_live, d_val, d_tempo = preset_values[preset]

        with st.form("cluster_form"):
            st.markdown("#### 🎧 Acoustic Feature Sliders")
            s1, s2, s3 = st.columns(3)
            with s1:
                danceability = st.slider("Danceability (Rhythm stability & beat strength)", 0.0, 1.0, float(d_dance), 0.01)
                energy = st.slider("Energy (Intensity, loudness, activity)", 0.0, 1.0, float(d_energy), 0.01)
                valence = st.slider("Valence (Musical positiveness / happiness)", 0.0, 1.0, float(d_val), 0.01)
            with s2:
                loudness = st.slider("Loudness (Overall sound level in dB)", -40.0, 0.0, float(d_loud), 0.5)
                tempo = st.slider("Tempo (Estimated pace in BPM)", 50.0, 220.0, float(d_tempo), 1.0)
                acousticness = st.slider("Acousticness (Confidence track is acoustic)", 0.0, 1.0, float(d_acoust), 0.01)
            with s3:
                instrumentalness = st.slider("Instrumentalness (Absence of vocal content)", 0.0, 1.0, float(d_instru), 0.01)
                speechiness = st.slider("Speechiness (Presence of spoken words)", 0.0, 1.0, float(d_speech), 0.01)
                liveness = st.slider("Liveness (Presence of an audience / live setting)", 0.0, 1.0, float(d_live), 0.01)

            st.divider()
            cb1, cb2, cb3 = st.columns([1, 2, 1])
            with cb2:
                cluster_submitted = st.form_submit_button("🔮 Discover Musical Persona & Cluster", use_container_width=True, type="primary")

        if cluster_submitted:
            raw_audio = {
                "danceability": danceability,
                "energy": energy,
                "loudness": loudness,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
            }

            try:
                audio_df = prepare_clustering_input(raw_audio)
                res = predict_cluster(cluster_bundle, audio_df)

                st.markdown("---")
                st.markdown("### Identified Musical Archetype & Persona")

                c_id = res["cluster_id"]
                p_name = res["persona_name"]
                desc = res["description"]

                st.markdown(
                    f"""
                    <div class="result-cluster">
                        <h2 style="margin:0; color:#4a148c;">🎵 Cluster {c_id}: {p_name}</h2>
                        <p style="margin:0.5rem 0 0 0; font-size:1.05rem; color:#212121;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Diagnostic Coordinates
                q1, q2, q3 = st.columns(3)
                with q1:
                    st.metric("Assigned Cluster ID", f"Cluster {c_id}")
                with q2:
                    st.metric("Euclidean Distance to Centroid", f"{res['distance_to_center']:.2f}")
                with q3:
                    st.metric("2D PCA Coordinate", f"({res['pca_x']:.2f}, {res['pca_y']:.2f})")

                # Sample Track Recommendations from Spotify Dataset
                if res["exemplars"]:
                    st.markdown("#### 🎶 Representative Songs in This Cluster (Spotify Dataset)")
                    st.dataframe(
                        pd.DataFrame(res["exemplars"])[["track_name", "artists", "album_name", "genre"]].rename(
                            columns={
                                "track_name": "Track Title",
                                "artists": "Artist",
                                "album_name": "Album",
                                "genre": "Genre",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as e:
                st.error(f"Clustering inference failed: {e}")

        # Transparency expander
        with st.expander("📈 Clustering Model Comparison & Validation Metrics", expanded=False):
            if cluster_metadata:
                eval_metrics = cluster_metadata.get("evaluation_metrics", {})
                km_met = eval_metrics.get("kmeans", {})
                h_met = eval_metrics.get("hierarchical", {})
                agr = eval_metrics.get("agreement", {})
                pca_v = eval_metrics.get("pca_variance_explained", {})

                st.markdown("#### Model Performance Comparison ($k=5$)")
                comp_df = pd.DataFrame([
                    {
                        "Algorithm": "K-Means Clustering",
                        "Silhouette Score (↑)": km_met.get("silhouette_score"),
                        "Davies-Bouldin Index (↓)": km_met.get("davies_bouldin_index"),
                        "Calinski-Harabasz Index (↑)": km_met.get("calinski_harabasz_index"),
                        "Complexity": "O(k·N·d) — Linear & Scalable",
                    },
                    {
                        "Algorithm": "Agglomerative Hierarchical",
                        "Silhouette Score (↑)": h_met.get("silhouette_score"),
                        "Davies-Bouldin Index (↓)": h_met.get("davies_bouldin_index"),
                        "Calinski-Harabasz Index (↑)": h_met.get("calinski_harabasz_index"),
                        "Complexity": "O(N² log N) — Dendrogram Hierarchical",
                    },
                ])
                st.table(comp_df)

                st.markdown("#### Partition Concordance & Manifold Variance")
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    st.metric("Adjusted Rand Index (ARI)", f"{agr.get('adjusted_rand_score', 0):.4f}")
                with cm2:
                    st.metric("Normalized Mutual Info (NMI)", f"{agr.get('normalized_mutual_info', 0):.4f}")
                with cm3:
                    st.metric("2D PCA Total Variance", f"{pca_v.get('total_pct', 0):.1f}%")

                st.markdown(
                    """
                    #### Methodology & Evaluation Notes
                    - **Elbow & Silhouette Selection:** Optimal $k=5$ determined at the major WCSS inflection point.
                    - **Dendrogram Cut:** Hierarchical clustering cut using Ward's minimum variance criterion.
                    - **Isotropic Scaling:** `StandardScaler` applied to prevent loudness (dB) and tempo (BPM) from biasing Euclidean space.
                    """
                )
