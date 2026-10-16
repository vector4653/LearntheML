"""
Bank Marketing Term Deposit Prediction - Streamlit Application
================================================================
Interactive GUI for the ML Capstone Project (Review 2 Bonus).
Predicts whether a bank client will subscribe to a term deposit
using a trained Gradient Boosting Classifier.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so `app.inference` resolves correctly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.inference import (
    CATEGORICAL_OPTIONS,
    load_metadata,
    load_model,
    predict,
    prepare_input,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
    }
    .result-yes {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-no {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
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
# Load model (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    """Load the trained pipeline once and cache it."""
    return load_model()


@st.cache_data(show_spinner=False)
def get_metadata():
    """Load model metadata once and cache it."""
    return load_metadata()


# ---------------------------------------------------------------------------
# Sidebar - project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/bank-building.png",
        width=80,
    )
    st.markdown("### About This Project")
    st.markdown(
        """
        **B.Tech ML Capstone Project**

        This application demonstrates a trained classification model
        that predicts whether a bank client will subscribe to a
        **term deposit** based on their demographic and campaign data.

        **Dataset:** UCI Bank Marketing Dataset (`bank-full.csv`)
        — 45,211 records, 16 features.

        **Model:** Gradient Boosting Classifier
        (scikit-learn Pipeline with StandardScaler + OneHotEncoder).
        """
    )
    st.divider()
    st.markdown("#### Team")
    st.markdown("Ritvik &bull; Abhisarg &bull; Rohan", unsafe_allow_html=True)
    st.divider()
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Disclaimer:</strong> This is an academic demonstration.
        Predictions are not financial advice and should not be used for
        real-world business decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🏦 Bank Marketing Term Deposit Prediction</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict whether a client will subscribe to a term deposit '
    "based on their profile and campaign interaction data.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Try loading the model
# ---------------------------------------------------------------------------
try:
    pipeline = get_model()
    metadata = get_metadata()
    model_loaded = True
except FileNotFoundError as e:
    st.error(
        f"**Model not found.** {e}\n\n"
        "Please run `python scripts/export_model.py` from the repository root to generate the model artifact."
    )
    model_loaded = False
except Exception as e:
    st.error(f"**Failed to load model:** {e}")
    model_loaded = False

if model_loaded:
    # -------------------------------------------------------------------
    # Input form
    # -------------------------------------------------------------------
    st.markdown("### Enter Client & Campaign Details")

    with st.form("prediction_form"):
        # --- Row 1: Customer Profile ---
        st.markdown("#### 👤 Customer Profile")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        with col2:
            job = st.selectbox("Job", options=CATEGORICAL_OPTIONS["job"], index=4)
        with col3:
            marital = st.selectbox("Marital Status", options=CATEGORICAL_OPTIONS["marital"], index=1)
        with col4:
            education = st.selectbox("Education", options=CATEGORICAL_OPTIONS["education"], index=1)

        # --- Row 2: Financial Information ---
        st.markdown("#### 💰 Financial Information")
        col5, col6, col7, col8 = st.columns(4)

        with col5:
            balance = st.number_input(
                "Average Yearly Balance (EUR)",
                min_value=-10000,
                max_value=200000,
                value=1000,
                step=100,
                help="Average yearly balance in euros.",
            )
        with col6:
            default = st.selectbox(
                "Has Credit Default?",
                options=["no", "yes"],
                index=0,
                help="Has the client defaulted on credit?",
            )
        with col7:
            housing = st.selectbox(
                "Has Housing Loan?",
                options=["no", "yes"],
                index=0,
            )
        with col8:
            loan = st.selectbox(
                "Has Personal Loan?",
                options=["no", "yes"],
                index=0,
            )

        # --- Row 3: Campaign Details ---
        st.markdown("#### 📞 Campaign Details")
        col9, col10, col11, col12 = st.columns(4)

        with col9:
            contact = st.selectbox(
                "Contact Type",
                options=CATEGORICAL_OPTIONS["contact"],
                index=0,
            )
        with col10:
            day = st.number_input(
                "Day of Month",
                min_value=1,
                max_value=31,
                value=15,
                step=1,
                help="Last contact day of the month.",
            )
        with col11:
            month = st.selectbox(
                "Month of Last Contact",
                options=CATEGORICAL_OPTIONS["month"],
                index=4,  # May
            )
        with col12:
            duration = st.number_input(
                "Last Call Duration (sec)",
                min_value=0,
                max_value=5000,
                value=180,
                step=10,
                help="Duration of the last contact call in seconds.",
            )

        # --- Row 4: Campaign History ---
        st.markdown("#### 📊 Campaign History")
        col13, col14, col15, col16 = st.columns(4)

        with col13:
            campaign = st.number_input(
                "Contacts During Campaign",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                help="Number of contacts during this campaign.",
            )
        with col14:
            pdays = st.number_input(
                "Days Since Last Contact",
                min_value=-1,
                max_value=999,
                value=-1,
                step=1,
                help="Days since last contact from a previous campaign (-1 = not contacted).",
            )
        with col15:
            previous = st.number_input(
                "Previous Campaign Contacts",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                help="Number of contacts before this campaign.",
            )
        with col16:
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                options=CATEGORICAL_OPTIONS["poutcome"],
                index=3,  # unknown
            )

        # --- Submit ---
        st.divider()
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button(
                "🔮 Predict Subscription",
                use_container_width=True,
                type="primary",
            )

    # -------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------
    if submitted:
        # Assemble raw input
        raw_input = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }

        try:
            input_df = prepare_input(raw_input)
            result = predict(pipeline, input_df)

            st.markdown("---")
            st.markdown("### Prediction Result")

            predicted = result["predicted_class"]
            probabilities = result["probabilities"]

            if predicted == "yes":
                st.markdown(
                    '<div class="result-yes">'
                    "<h3>✅ Likely to Subscribe</h3>"
                    "<p>The model predicts this client is <strong>likely</strong> to subscribe "
                    "to a term deposit.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-no">'
                    "<h3>❌ Unlikely to Subscribe</h3>"
                    "<p>The model predicts this client is <strong>unlikely</strong> to subscribe "
                    "to a term deposit.</p></div>",
                    unsafe_allow_html=True,
                )

            # Probabilities
            if probabilities:
                st.markdown("#### Predicted Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(
                        label="P(No Subscription)",
                        value=f"{probabilities['no']:.1%}",
                    )
                with prob_col2:
                    st.metric(
                        label="P(Subscription)",
                        value=f"{probabilities['yes']:.1%}",
                    )
                st.progress(probabilities["yes"])
                st.caption(
                    "**Note:** These are model-estimated probabilities, not guaranteed outcomes. "
                    "They reflect patterns learned from historical data."
                )

            st.info(f"**Model used:** {result['model_name']}")

        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.exception(e)

    # -------------------------------------------------------------------
    # Model performance section (always visible)
    # -------------------------------------------------------------------
    with st.expander("📈 Model Performance & Transparency", expanded=False):
        if metadata:
            metrics = metadata.get("evaluation_metrics", {})
            model_name = metadata.get("model_name", "N/A")
            params = metadata.get("model_params", {})
            versions = metadata.get("versions", {})

            st.markdown(f"**Selected Model:** {model_name}")
            st.markdown(
                "**Selection Rationale:** Gradient Boosting Classifier was selected as the "
                "deployment model based on its strong performance across accuracy, weighted F1-score, "
                "and ROC-AUC on the held-out test set. It is one of the ten classification algorithms "
                "evaluated in the capstone project (Review 2, Algorithm #8)."
            )

            st.markdown("#### Evaluation Metrics (Test Set)")
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 'N/A'):.2%}" if isinstance(metrics.get('accuracy'), (int, float)) else "N/A")
            with m_col2:
                st.metric("Precision (W)", f"{metrics.get('weighted_precision', 'N/A'):.2%}" if isinstance(metrics.get('weighted_precision'), (int, float)) else "N/A")
            with m_col3:
                st.metric("Recall (W)", f"{metrics.get('weighted_recall', 'N/A'):.2%}" if isinstance(metrics.get('weighted_recall'), (int, float)) else "N/A")
            with m_col4:
                st.metric("F1-Score (W)", f"{metrics.get('weighted_f1', 'N/A'):.2%}" if isinstance(metrics.get('weighted_f1'), (int, float)) else "N/A")
            with m_col5:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(metrics.get('roc_auc'), (int, float)) else "N/A")

            st.markdown("#### Model Configuration")
            st.json(params)

            st.markdown("#### Environment Versions")
            st.json(versions)
        else:
            st.warning("Model metadata not available. Run the export script to generate it.")

        st.markdown(
            """
            #### Methodology Notes
            - **Preprocessing:** StandardScaler for numerical features, OneHotEncoder (drop='first') for categorical features.
            - **Feature Engineering:** `pdays_contacted`, `total_contacts`, `is_retired_or_student`, `duration_min`.
            - **Data Split:** 80/20 stratified train-test split (random_state=42).
            - **Leakage Prevention:** All transformers fitted exclusively on training data.
            - **Dataset:** UCI Bank Marketing Dataset (45,211 records).
            """
        )

    # -------------------------------------------------------------------
    # Sample inputs section
    # -------------------------------------------------------------------
    with st.expander("🧪 Try Sample Inputs", expanded=False):
        st.markdown(
            """
            **Suggested test cases to try:**

            | Scenario | Age | Job | Balance | Duration | poutcome | Expected |
            |---|---|---|---|---|---|---|
            | High-value retired | 65 | retired | 15000 | 500 | success | Likely Yes |
            | Young student | 22 | student | 500 | 300 | unknown | Possibly Yes |
            | Short call, low engagement | 40 | blue-collar | 200 | 30 | failure | Likely No |
            | Multiple campaigns, no response | 55 | management | 1500 | 60 | failure | Likely No |

            *Adjust the form fields above and click Predict to test.*
            """
        )
