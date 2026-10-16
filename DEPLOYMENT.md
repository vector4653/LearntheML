# Deployment Guide — Dual-Track Machine Learning Application

## Hosting Platform
**Streamlit Community Cloud** (free tier) — https://share.streamlit.io

## Prerequisites
1. The repository must be **public** on GitHub (or you must grant repository access to Streamlit Cloud).
2. A **Streamlit Community Cloud account** linked to the GitHub account owning `vector4653/LearntheML`.
3. Both serialized model artifacts must be committed and pushed to the repository:
   - `models/best_classification_pipeline.joblib` (~869 KB)
   - `models/best_regression_pipeline.joblib` (~551 KB)

---

## Deployment Steps

### 1. Generate Model Artifacts (if not already committed)
```bash
cd LearntheML
python scripts/export_model.py             # Generates Bank Marketing classification pipeline
python scripts/export_regression_model.py  # Generates Seoul Bike regression pipeline
```

### 2. Commit and Push All Changes
```bash
git add .
git commit -m "feat: dual-track Streamlit app with classification and regression models"
git push origin main
```

### 3. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"New app"**.
3. Configure the deployment settings:
   - **Repository:** `vector4653/LearntheML`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Click **"Deploy!"**.
5. Streamlit Cloud will automatically provision a Python container, install dependencies from `requirements.txt`, and start the app (typically 2-4 minutes).

### 4. Verify the Deployment
- Open your live public URL (e.g., `https://learntheml-xxxx.streamlit.app`).
- Test **Track 1 (Bank Marketing)**:
  - Submit the form $\rightarrow$ Verify class prediction (`yes`/`no`) and probability metrics.
- Switch via sidebar to **Track 2 (Seoul Bike Sharing)**:
  - Adjust hour, season, and temperature $\rightarrow$ Verify hourly rented bike count prediction.
- Confirm both **Model Performance & Transparency** sections display verified metrics without error.

---

## Runtime Configuration

### Dependency Specifications (`requirements.txt`)
- `scikit-learn==1.9.1` — **Strictly pinned** to match the environment used to train and serialize the pipelines.
- `streamlit>=1.30.0`
- `joblib>=1.3.0`
- Core analytics: `numpy>=1.26.0`, `pandas>=2.0.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`

### Streamlit Settings (`.streamlit/config.toml`)
Configured for headless cloud deployment with custom university/project branding theme.

---

## Updating the Deployed Application
When you push new commits to the `main` branch, Streamlit Cloud detects the update and triggers automatic continuous deployment.

To manually reboot or view logs:
1. Access your application dashboard on [share.streamlit.io](https://share.streamlit.io).
2. Click **"Manage app"** $\rightarrow$ **"Logs"** to inspect live runtime logs or trigger **"Reboot app"**.

---

## Troubleshooting

### Missing Model Artifacts
```
FileNotFoundError: Model artifact not found at models/...
```
**Resolution:** Re-run the export scripts and commit the resulting `.joblib` files:
```bash
git add models/
git commit -m "fix: ensure model artifacts are tracked"
git push origin main
```

### Scikit-Learn Incompatible Version Error
```
ValueError: Trying to unpickle estimator ... from version ...
```
**Resolution:** Ensure `requirements.txt` has `scikit-learn==1.9.1`. Both models were exported using scikit-learn 1.9.1.
