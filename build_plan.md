# Capstone Review 1 Build Plan

## Objective
Deliver an end-to-end Machine Learning pipeline for the Regression and Classification (Part A) tracks prior to mid-semester examinations.

## Architecture
* **Environment:** Python 3, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
* **Repository Structure:** `/data`, `/notebooks`, `/notebooks/py_files`, `README.md`, `requirements.txt`.
* **Development Workflow (Script-First):** 
  * All active modeling, feature engineering, and pipeline logic must first be authored, verified, and run as modular scripts in `/notebooks/py_files/` (`.py` files).
  * Once validated via script execution, code is transferred into the respective Jupyter notebooks (`/notebooks/*.ipynb`).
  * Each notebook code cell must be accompanied by preceding Markdown documentation clearly explaining what the code block below does, its mathematical/engineering rationale, and key observations.
* **Seed:** `random_state=42` enforced globally.

## Phased Rollout
1. **Environment Setup:** Init repo structure, requirements, standalone `.py` scripts, and empty notebooks.
2. **EDA:** Audit data, plot distributions, heatmaps, and relationships with Markdown insights.
3. **Preprocessing:** Clean missing values, engineer features, encode, scale (fit on train only), and split.
4. **Regression (Seoul Bike):** Train 10 algorithms, tune 2 via Grid/RandomizedSearchCV, plot residuals/feature importance, and output an R²-ranked table.
5. **Classification Part A (Bank Marketing):** Train 5 baseline algorithms, output Accuracy/F1/Confusion Matrices, and build a preliminary comparison table.
6. **Integration Check:** Verify top-to-bottom execution with zero errors across both `.py` files and notebooks, and execute Git commits.
