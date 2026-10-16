# Deployment Guide - Bank Marketing Prediction Application

## Hosting Platform
**Streamlit Community Cloud** (free tier) — https://share.streamlit.io

## Prerequisites
1. The repository must be **public** on GitHub (or you must grant access to Streamlit Cloud).
2. A **Streamlit Community Cloud account** linked to the GitHub account that owns the repository.
3. The model artifact `models/best_classification_pipeline.joblib` must be committed and pushed to the repository.

## Deployment Steps

### 1. Generate the Model Artifact (if not already present)
```bash
cd LearntheML
python scripts/export_model.py
```
This creates:
- `models/best_classification_pipeline.joblib` — serialized sklearn Pipeline
- `models/model_metadata.json` — evaluation metrics and schema

### 2. Commit and Push All Changes
```bash
git add .
git commit -m "feat: add Streamlit GUI and model artifact for Review 2 bonus"
git push origin main
```

### 3. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"New app"**.
3. Configure:
   - **Repository:** `vector4653/LearntheML`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Click **"Deploy!"**.
5. Wait 2-5 minutes for the build to complete.

### 4. Verify Deployment
- Open the generated public URL.
- Fill in the prediction form and click **Predict Subscription**.
- Confirm the prediction result and probabilities are displayed.
- Check the **Model Performance & Transparency** section.

## Configuration

### requirements.txt
The `requirements.txt` at the repository root contains all dependencies. Streamlit Cloud automatically installs these. Key pinned versions:
- `scikit-learn==1.9.1` — **must match** the version used to serialize the model
- `streamlit>=1.30.0`
- `joblib>=1.3.0`

### Python Version
If you need to pin the Python version, create a `.python-version` file at the repo root:
```
3.11
```

### No Secrets Required
This application does not use external APIs, databases, or credentials.

## Updating the Deployed Application
After pushing changes to the `main` branch on GitHub, Streamlit Cloud will automatically redeploy within a few minutes.

To manually trigger a redeploy:
1. Go to your app's management page on share.streamlit.io.
2. Click **"Reboot app"** in the top-right menu (⋮).

## Troubleshooting

### Model not found error
```
FileNotFoundError: Model artifact not found at models/best_classification_pipeline.joblib
```
**Fix:** Ensure the model file is committed and pushed:
```bash
git add models/best_classification_pipeline.joblib models/model_metadata.json
git commit -m "add model artifact"
git push origin main
```

### scikit-learn version mismatch
```
ValueError: Incompatible sklearn version...
```
**Fix:** Ensure `requirements.txt` pins `scikit-learn==1.9.1` (the exact version used during `export_model.py`).

### Import errors / ModuleNotFoundError
**Fix:** Verify all required packages are listed in `requirements.txt`. Streamlit Cloud does not install from `conda` — only `pip` packages.

### App crashes on startup
Check the Streamlit Cloud logs:
1. Go to your app's management page.
2. Click **"Manage app"** → **"Logs"**.

### Large model file (>100 MB)
The current model is ~870 KB, well within limits. If future models exceed GitHub's 100 MB file limit, use [Git LFS](https://git-lfs.github.com/).

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Generate model (first time only)
python scripts/export_model.py

# Run the app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.
