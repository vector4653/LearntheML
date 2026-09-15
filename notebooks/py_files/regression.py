#!/usr/bin/env python
# coding: utf-8

# # Seoul Bike Sharing Demand Prediction - Regression Track
# 
# ## Capstone Project - Review 1
# 
# **Objective:** Predict the hourly count of rented bikes in Seoul based on weather, environmental, and calendar features.
# 
# **Reproducibility:** Global `random_state=42` is strictly enforced.
# 
# **Leakage Prevention:** All transformations, scalers, and encoders are strictly fitted on the training split only.

# In[ ]:


# Core Imports & Global Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Set LOKY_MAX_CPU_COUNT to prevent Windows joblib wmic core-detection warning
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 4))

# Reproducibility seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

def get_data_path(rel_path):
    candidates = [
        rel_path,
        os.path.join('..', rel_path) if not rel_path.startswith('..') else rel_path,
        os.path.join('../..', rel_path.lstrip('./\\')),
        os.path.join('data', os.path.basename(rel_path)),
        os.path.join('../data', os.path.basename(rel_path)),
        os.path.join('../../data', os.path.basename(rel_path)),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path

print("Regression notebook initialized. Random state set to 42.")


# ## Phase 2: Exploratory Data Analysis (EDA)
# 
# In this phase, we perform an exhaustive audit of the Seoul Bike Sharing dataset:
# 1. **Data Ingestion & Integrity Audit:** Clean headers, inspect data types, detect missing values.
# 2. **Target Variable Analysis (`rented_bike_count`):** Distribution shape, skewness, zero-demand hours.
# 3. **Temporal & Seasonal Patterns:** Hourly peak commuter trends, day/night cycles, and seasonal shifts.
# 4. **Environmental & Weather Dynamics:** Correlation with temperature, humidity, solar radiation, and rainfall.
# 5. **Multicollinearity Screening:** Feature-to-feature correlation analysis.

# In[ ]:


# 1. Load Dataset and Standardize Column Headers
raw_bike_df = pd.read_csv(get_data_path('data/SeoulBikeData.csv'), encoding='unicode_escape')

# Clean column names to standard lower_snake_case without special characters
bike_df = raw_bike_df.copy()
bike_df.columns = [
    'date',
    'rented_bike_count',
    'hour',
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall',
    'seasons',
    'holiday',
    'functioning_day'
]

print(f"Dataset Shape: {bike_df.shape[0]} rows, {bike_df.shape[1]} columns")
bike_df.head()


# In[ ]:


# 2. Data Integrity & Completeness Audit
print("=== Column Schema & Data Types ===")
print(bike_df.info())

print("\n=== Missing Values Check ===")
missing_counts = bike_df.isnull().sum()
print(missing_counts[missing_counts > 0] if missing_counts.sum() > 0 else "No missing values found across all columns.")

print("\n=== Statistical Summary (Numerical Features) ===")
bike_df.describe().T


# In[ ]:


# 3. Categorical Variables Audit
cat_cols = ['seasons', 'holiday', 'functioning_day']
for col in cat_cols:
    print(f"--- Value Counts: {col} ---")
    print(bike_df[col].value_counts())
    print()


# In[ ]:


# 4. Target Variable Distribution (rented_bike_count)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram & KDE
sns.histplot(bike_df['rented_bike_count'], kde=True, ax=axes[0], color='#2b5c8f', bins=40)
axes[0].axvline(bike_df['rented_bike_count'].mean(), color='red', linestyle='--', label=f"Mean: {bike_df['rented_bike_count'].mean():.1f}")
axes[0].axvline(bike_df['rented_bike_count'].median(), color='green', linestyle='-', label=f"Median: {bike_df['rented_bike_count'].median():.1f}")
axes[0].set_title('Target Distribution: Rented Bike Count')
axes[0].set_xlabel('Rented Bike Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Boxplot to inspect extreme values
sns.boxplot(x=bike_df['rented_bike_count'], ax=axes[1], color='#6baed6')
axes[1].set_title('Boxplot: Rented Bike Count (Outlier Inspection)')
axes[1].set_xlabel('Rented Bike Count')

plt.tight_layout()
plt.show()

# Skewness calculation
target_skew = bike_df['rented_bike_count'].skew()
print(f"Target Skewness: {target_skew:.3f} (Right-skewed, heavy right tail)")


# In[ ]:


# 5. Temporal Dynamics: Hourly Rental Patterns
fig, ax = plt.subplots(figsize=(12, 5))
hourly_avg = bike_df.groupby('hour')['rented_bike_count'].agg(['mean', 'std']).reset_index()

sns.lineplot(data=bike_df, x='hour', y='rented_bike_count', ci=95, ax=ax, color='#1f77b4', marker='o')
ax.set_title('Average Hourly Bike Rentals (With 95% Confidence Interval)')
ax.set_xlabel('Hour of the Day (0 - 23)')
ax.set_ylabel('Mean Rented Bike Count')
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle='--', alpha=0.6)

# Annotate commuter rush hours
ax.axvspan(7.5, 9.5, color='orange', alpha=0.2, label='Morning Peak (8-9 AM)')
ax.axvspan(17.5, 19.5, color='red', alpha=0.2, label='Evening Peak (6-7 PM)')
ax.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# 6. Seasonal & Calendar Variations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Rentals across Seasons
sns.boxplot(data=bike_df, x='seasons', y='rented_bike_count', ax=axes[0], palette='Set2')
axes[0].set_title('Demand across Seasons')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Rented Bike Count')

# Rentals on Holidays vs Regular Days
sns.boxplot(data=bike_df, x='holiday', y='rented_bike_count', ax=axes[1], palette='Pastel1')
axes[1].set_title('Demand: Holiday vs Non-Holiday')
axes[1].set_xlabel('Holiday Status')
axes[1].set_ylabel('Rented Bike Count')

# Rentals on Functioning vs Non-Functioning Days
sns.barplot(data=bike_df, x='functioning_day', y='rented_bike_count', ax=axes[2], palette='coolwarm')
axes[2].set_title('Demand: Functioning vs Non-Functioning Day')
axes[2].set_xlabel('Functioning Day')
axes[2].set_ylabel('Mean Rented Bike Count')

plt.tight_layout()
plt.show()


# In[ ]:


# 7. Weather & Environmental Relationships
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

weather_vars = [
    ('temperature', 'Temperature (°C)', '#d95f02'),
    ('humidity', 'Humidity (%)', '#7570b3'),
    ('wind_speed', 'Wind Speed (m/s)', '#1b9e77'),
    ('solar_radiation', 'Solar Radiation (MJ/m2)', '#e7298a'),
    ('rainfall', 'Rainfall (mm)', '#e6ab02'),
    ('snowfall', 'Snowfall (cm)', '#66a61e')
]

for idx, (col, label, color) in enumerate(weather_vars):
    ax = axes[idx // 3, idx % 3]
    sns.scatterplot(data=bike_df.sample(2000, random_state=RANDOM_STATE), x=col, y='rented_bike_count', alpha=0.3, ax=ax, color=color)
    sns.regplot(data=bike_df, x=col, y='rented_bike_count', scatter=False, ax=ax, color='black', line_kws={'linewidth': 1.5})
    ax.set_title(f'Rentals vs {label}')
    ax.set_xlabel(label)
    ax.set_ylabel('Rented Bike Count')

plt.tight_layout()
plt.show()


# In[ ]:


# 8. Pearson Correlation Heatmap & Multicollinearity Inspection
numeric_cols = bike_df.select_dtypes(include=[np.number]).columns
corr_matrix = bike_df[numeric_cols].corr()

plt.figure(figsize=(11, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, mask=mask, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Pearson Correlation Matrix (Numerical Features)', fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# Highlight highest absolute correlations with Target and between features
print("=== Correlations with Rented Bike Count ===")
print(corr_matrix['rented_bike_count'].sort_values(ascending=False))

print("\n=== High Feature-to-Feature Collinearity (|r| > 0.70) ===")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
        if col1 != 'rented_bike_count' and col2 != 'rented_bike_count':
            r = corr_matrix.loc[col1, col2]
            if abs(r) > 0.70:
                print(f"Collinear Pair: {col1} <--> {col2} (r = {r:.3f})")


# ### Phase 2 Key EDA Findings & Modeling Implications
# 
# 1. **Target Distribution (rented_bike_count):**
#    * Heavily right-skewed (skewness ≈ 1.15) with positive median-mean divergence (median = 504.5, mean = 704.6).
#    * Non-functioning days (functioning_day == 'No') record strictly 0 rentals across all hours. This is a deterministic structural filter that will be handled appropriately during preprocessing.
# 2. **Bimodal Commuter Rhythm:**
#    * Demand exhibits sharp bimodal spikes aligning with workplace transit: morning rush hour at **08:00** and evening rush hour at **18:00**.
#    * Low demand is observed during early morning hours (03:00 - 05:00).
# 3. **Seasonal & Calendar Effects:**
#    * Summer and Autumn have the highest median rental counts, whereas Winter shows steep drops due to sub-zero temperatures and snow.
# 4. **Multicollinearity & Feature Redundancy:**
#    * Temperature and dew_point_temperature exhibit extremely high collinearity (r ≈ 0.91). Keeping both in unregularized linear models risks variance inflation, highlighting the importance of Ridge/Lasso regularization and tree-based methods in Phase 4.


# In[ ]:


# ==============================================================================
# PHASE 3: PREPROCESSING & FEATURE ENGINEERING (@Abhisarg)
# ==============================================================================
# Author / Assignee: @Abhisarg
# Track: Seoul Bike Sharing Demand Prediction (Regression)
# 
# Pipeline Summary:
# 1. Missing Value Strategy: Dataset verified with 0 NaN entries. Outlier filter: non-functioning days removed.
# 2. Feature Engineering: Engineered day_of_week, is_weekend, rush_hour, and temp_humidity_idx.
# 3. Train-Test Split: 80/20 split using global random_state=42.
# 4. Leakage-Free Scaling & Encoding: StandardScaler and OneHotEncoder fitted exclusively on X_train.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("=" * 78)
print("PHASE 3: SEOUL BIKE PREPROCESSING & FEATURE ENGINEERING (@Abhisarg)")
print("=" * 78)

# --- Step 1: Ingestion & Column Normalization ---
bike_df = pd.read_csv(get_data_path('data/SeoulBikeData.csv'), encoding='unicode_escape')
bike_df.columns = [
    'date', 'rented_bike_count', 'hour', 'temperature', 'humidity',
    'wind_speed', 'visibility', 'dew_point_temperature', 'solar_radiation',
    'rainfall', 'snowfall', 'seasons', 'holiday', 'functioning_day'
]

# --- Step 2: Missing Value Strategy & Structural Outlier Filter ---
print("=== Missing Values Audit ===")
print(bike_df.isnull().sum())

# Remove non-functioning hours where rentals are deterministically 0
bike_clean = bike_df[bike_df['functioning_day'] == 'Yes'].copy()
print(f"\nRows after filtering non-functioning days: {len(bike_clean)} (Removed {len(bike_df) - len(bike_clean)} non-functioning hours)")

# --- Step 3: Feature Engineering (@Abhisarg) ---
bike_clean['datetime'] = pd.to_datetime(bike_clean['date'], format='%d/%m/%Y')
bike_clean['day_of_week'] = bike_clean['datetime'].dt.dayofweek
bike_clean['is_weekend'] = bike_clean['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
bike_clean['rush_hour'] = bike_clean.apply(
    lambda row: 1 if (row['is_weekend'] == 0 and row['hour'] in [7, 8, 9, 17, 18, 19]) else 0,
    axis=1
)
bike_clean['temp_humidity_idx'] = bike_clean['temperature'] * (bike_clean['humidity'] / 100.0)
print("\nEngineered Features Added: ['day_of_week', 'is_weekend', 'rush_hour', 'temp_humidity_idx']")

# --- Step 4: Define Features and Target ---
X_bike = bike_clean.drop(columns=['date', 'datetime', 'rented_bike_count', 'functioning_day'])
y_bike = bike_clean['rented_bike_count']

cat_cols_bike = ['seasons', 'holiday']
num_cols_bike = [col for col in X_bike.columns if col not in cat_cols_bike]

# --- Step 5: Reproducible Train-Test Split (random_state=42) ---
X_train_bike, X_test_bike, y_train_bike, y_test_bike = train_test_split(
    X_bike, y_bike, test_size=0.20, random_state=RANDOM_STATE
)
print(f"\nTrain-Test Split Completed (random_state=42):")
print(f"- X_train shape: {X_train_bike.shape}")
print(f"- X_test shape:  {X_test_bike.shape}")

# --- Step 6: Leakage-Free Encoding & Scaling (Fitted Exclusively on X_train) ---
preprocessor_bike = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols_bike),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols_bike)
    ]
)

X_train_bike_scaled = preprocessor_bike.fit_transform(X_train_bike)
X_test_bike_scaled = preprocessor_bike.transform(X_test_bike)

cat_feature_names = preprocessor_bike.named_transformers_['cat'].get_feature_names_out(cat_cols_bike)
all_feature_names_bike = num_cols_bike + list(cat_feature_names)

print(f"\nProcessed Training Feature Matrix (Scaled & Encoded): {X_train_bike_scaled.shape}")
print(f"Processed Testing Feature Matrix (Scaled & Encoded):  {X_test_bike_scaled.shape}")
print(f"Total Scaled/Encoded Features: {len(all_feature_names_bike)}")
print(f"Feature Names: {all_feature_names_bike}")
# ==============================================================================


# In[ ]:


# ==============================================================================
# PHASE 4: REGRESSION TRACK - MODELING PART A (@Abhisarg)
# ==============================================================================
# Author / Assignee: @Abhisarg
# Models Implemented:
# 1. Linear Regression (Ordinary Least Squares)
# 2. Ridge Regression (L2 Regularization)
# 3. Lasso Regression (L1 Regularization & Feature Selection)
# 4. ElasticNet Regression (L1 + L2 Hybrid Regularization)
# 5. Polynomial Regression (Degree 2 Interaction Terms + Regularization)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

print("=" * 78)
print("PHASE 4: REGRESSION TRACK - MODELING PART A (@Abhisarg)")
print("=" * 78)

# Helper function to compute and log regression metrics
def evaluate_regression_model(name, model, X_tr, y_tr, X_te, y_te, best_params=None):
    # Fit model on training split if not already fitted via GridSearchCV
    if hasattr(model, 'best_estimator_'):
        best_est = model.best_estimator_
    else:
        model.fit(X_tr, y_tr)
        best_est = model
    
    # Predict on train and test splits
    y_tr_pred = best_est.predict(X_tr)
    y_te_pred = best_est.predict(X_te)
    
    # Calculate metrics
    tr_r2 = r2_score(y_tr, y_tr_pred)
    te_r2 = r2_score(y_te, y_te_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
    mae = mean_absolute_error(y_te, y_te_pred)
    
    res = {
        'Model': name,
        'Train R²': round(tr_r2, 4),
        'Test R²': round(te_r2, 4),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2)
    }
    if best_params:
        res['Tuned Hyperparameters'] = str(best_params)
    elif "OLS" in name:
        res['Tuned Hyperparameters'] = 'None (OLS analytical solution)'
    else:
        res['Tuned Hyperparameters'] = 'Default baseline'
        
    return res, y_te_pred

phase4_results = []
phase4_predictions = {}

# --- 1. Linear Regression (OLS Baseline) ---
lr_model = LinearRegression()
lr_res, lr_preds = evaluate_regression_model("Linear Regression (OLS)", lr_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(lr_res)
phase4_predictions["Linear Regression (OLS)"] = lr_preds

# --- 2. Ridge Regression (L2 Baseline, alpha=1.0) ---
ridge_base = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge_res, ridge_preds = evaluate_regression_model("Ridge Regression (Baseline)", ridge_base, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(ridge_res)
phase4_predictions["Ridge Regression (Baseline)"] = ridge_preds

# --- 3. Lasso Regression (L1 Baseline, alpha=0.1) ---
lasso_base = Lasso(alpha=0.1, random_state=RANDOM_STATE)
lasso_res, lasso_preds = evaluate_regression_model("Lasso Regression (Baseline)", lasso_base, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(lasso_res)
phase4_predictions["Lasso Regression (Baseline)"] = lasso_preds

# --- 4. ElasticNet Regression (Hybrid Baseline, alpha=0.1, l1_ratio=0.5) ---
elastic_base = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE)
elastic_res, elastic_preds = evaluate_regression_model("ElasticNet Regression (Baseline)", elastic_base, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(elastic_res)
phase4_predictions["ElasticNet Regression (Baseline)"] = elastic_preds

# --- 5. Polynomial Regression (Degree 2 Baseline, Ridge alpha=10.0) ---
poly_base = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=10.0, random_state=RANDOM_STATE))
])
poly_res, poly_preds = evaluate_regression_model("Polynomial Regression Degree 2 (Baseline)", poly_base, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(poly_res)
phase4_predictions["Polynomial Regression Degree 2 (Baseline)"] = poly_preds

# Summary Table (Linear Baselines 1-5)
linear_df = pd.DataFrame(phase4_results).sort_values(by='Test R²', ascending=False).reset_index(drop=True)
print("\n=== Phase 4 Model Evaluation Summary (Linear Models 1-5 Baseline) ===")
print(linear_df.to_string(index=False))
# ==============================================================================


# ==============================================================================
# PHASE 4: REGRESSION TRACK - MODELING PART B (@Ritvik Non-Linear Baselines)
# ==============================================================================
# Author / Assignee: @Ritvik
# Non-Linear Models Implemented:
# 6. Decision Tree Regressor (Non-parametric baseline)
# 7. Random Forest Regressor (Ensemble Bagging baseline, n_estimators=100)
# 8. Gradient Boosting Regressor (Ensemble Boosting baseline, n_estimators=100)
# 9. Support Vector Regressor (SVR with RBF kernel, C=1000.0, epsilon=10.0)
# 10. K-Nearest Neighbors Regressor (KNN distance-weighted, n_neighbors=5)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

print("\n" + "=" * 78)
print("PHASE 4: REGRESSION TRACK - MODELING PART B (@Ritvik Non-Linear Models)")
print("=" * 78)

# --- 6. Decision Tree Regressor ---
dt_model = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt_res, dt_preds = evaluate_regression_model("Decision Tree (Baseline)", dt_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(dt_res)
phase4_predictions["Decision Tree (Baseline)"] = dt_preds
print(f"Decision Tree Trained: Test R² = {dt_res['Test R²']:.4f}, RMSE = {dt_res['RMSE']:.2f}")

# --- 7. Random Forest Regressor ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_res, rf_preds = evaluate_regression_model("Random Forest (Baseline)", rf_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(rf_res)
phase4_predictions["Random Forest (Baseline)"] = rf_preds
print(f"Random Forest Trained: Test R² = {rf_res['Test R²']:.4f}, RMSE = {rf_res['RMSE']:.2f}")

# --- 8. Gradient Boosting Regressor ---
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
gb_res, gb_preds = evaluate_regression_model("Gradient Boosting (Baseline)", gb_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(gb_res)
phase4_predictions["Gradient Boosting (Baseline)"] = gb_preds
print(f"Gradient Boosting Trained: Test R² = {gb_res['Test R²']:.4f}, RMSE = {gb_res['RMSE']:.2f}")

# --- 9. Support Vector Regressor (SVR - RBF Kernel) ---
svr_model = SVR(kernel='rbf', C=1000.0, epsilon=10.0)
svr_res, svr_preds = evaluate_regression_model("Support Vector Regressor (SVR)", svr_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(svr_res)
phase4_predictions["Support Vector Regressor (SVR)"] = svr_preds
print(f"SVR Trained: Test R² = {svr_res['Test R²']:.4f}, RMSE = {svr_res['RMSE']:.2f}")

# --- 10. K-Nearest Neighbors Regressor (KNN) ---
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_res, knn_preds = evaluate_regression_model("K-Nearest Neighbors (KNN)", knn_model, X_train_bike_scaled, y_train_bike, X_test_bike_scaled, y_test_bike)
phase4_results.append(knn_res)
phase4_predictions["K-Nearest Neighbors (KNN)"] = knn_preds
print(f"KNN Trained: Test R² = {knn_res['Test R²']:.4f}, RMSE = {knn_res['RMSE']:.2f}")
# ==============================================================================


# ==============================================================================
# PHASE 4: REGRESSION TRACK - PART C: HYPERPARAMETER TUNING (Point 3)
# ==============================================================================
# Systematic 5-fold cross-validation tuning for:
# Model A (Linear Track): Ridge Regression via GridSearchCV
# Model B (Non-Linear Track): Random Forest Regressor via RandomizedSearchCV

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

print("\n" + "=" * 78)
print("PHASE 4: HYPERPARAMETER TUNING (1 Linear + 1 Non-Linear via 5-Fold CV)")
print("=" * 78)

# --- Model A (Linear): Ridge Regression 5-Fold GridSearchCV ---
print("1. Tuning Ridge Regression across regularization parameter alpha...")
ridge_param_grid = {'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]}
ridge_cv = GridSearchCV(Ridge(random_state=RANDOM_STATE), ridge_param_grid, cv=5, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_train_bike_scaled, y_train_bike)

ridge_tuned_res, ridge_tuned_preds = evaluate_regression_model(
    "Ridge Regression (Tuned - GridSearchCV)",
    ridge_cv,
    X_train_bike_scaled,
    y_train_bike,
    X_test_bike_scaled,
    y_test_bike,
    best_params=ridge_cv.best_params_
)
phase4_results.append(ridge_tuned_res)
phase4_predictions["Ridge Regression (Tuned)"] = ridge_tuned_preds

print(f"   -> Best Ridge alpha: {ridge_cv.best_params_['alpha']} (5-fold CV R² = {ridge_cv.best_score_:.4f})")
print(f"   -> Test R²: {ridge_tuned_res['Test R²']:.4f} (Baseline was {ridge_res['Test R²']:.4f})")

# --- Model B (Non-Linear): Random Forest 5-Fold RandomizedSearchCV ---
print("\n2. Tuning Random Forest Regressor across ensemble depth & tree structure...")
rf_param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 15, 20, 25],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 1.0]
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_dist,
    n_iter=12,
    cv=5,
    scoring='r2',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_search.fit(X_train_bike_scaled, y_train_bike)

rf_tuned_res, rf_tuned_preds = evaluate_regression_model(
    "Random Forest (Tuned - RandomizedSearchCV)",
    rf_search,
    X_train_bike_scaled,
    y_train_bike,
    X_test_bike_scaled,
    y_test_bike,
    best_params=rf_search.best_params_
)
phase4_results.append(rf_tuned_res)
phase4_predictions["Random Forest (Tuned)"] = rf_tuned_preds

print(f"   -> Best Random Forest params: {rf_search.best_params_} (5-fold CV R² = {rf_search.best_score_:.4f})")
print(f"   -> Test R²: {rf_tuned_res['Test R²']:.4f} (Baseline was {rf_res['Test R²']:.4f}, RMSE = {rf_tuned_res['RMSE']:.2f})")

# --- Direct Tuning Comparison Table ---
tuning_comp_df = pd.DataFrame([
    {
        'Family': 'Linear (L2)',
        'Model': 'Ridge Regression',
        'Baseline Test R²': ridge_res['Test R²'],
        'Tuned Test R²': ridge_tuned_res['Test R²'],
        'Baseline RMSE': ridge_res['RMSE'],
        'Tuned RMSE': ridge_tuned_res['RMSE'],
        'Selected Hyperparameters': str(ridge_cv.best_params_)
    },
    {
        'Family': 'Non-Linear (Bagging)',
        'Model': 'Random Forest',
        'Baseline Test R²': rf_res['Test R²'],
        'Tuned Test R²': rf_tuned_res['Test R²'],
        'Baseline RMSE': rf_res['RMSE'],
        'Tuned RMSE': rf_tuned_res['RMSE'],
        'Selected Hyperparameters': str(rf_search.best_params_)
    }
])
print("\n=== Direct Tuning Comparison: Baseline vs. Tuned ===")
print(tuning_comp_df.to_string(index=False))
# ==============================================================================


# ==============================================================================
# PHASE 4: REGRESSION TRACK - PART D: MASTER SUMMARY TABLE (Point 4)
# ==============================================================================
master_df = pd.DataFrame(phase4_results).sort_values(by='Test R²', ascending=False).reset_index(drop=True)
master_df['Rank'] = range(1, len(master_df) + 1)
cols_ordered = ['Rank', 'Model', 'Train R²', 'Test R²', 'RMSE', 'MAE', 'Tuned Hyperparameters']
master_df = master_df[cols_ordered]

print("\n" + "=" * 78)
print("=== Phase 4 Master Model Evaluation Summary (All Models Ranked by Test R²) ===")
print("=" * 78)
print(master_df.to_string(index=False))
# ==============================================================================


# ==============================================================================
# PHASE 4: REGRESSION TRACK - PART E: VISUAL DIAGNOSTICS (Point 5)
# ==============================================================================
# Diagnostics Generated:
# 1. Predicted vs. Actual Demand Scatter Plot (Tuned Random Forest)
# 2. Residual Diagnostics: Residuals vs. Fitted & Residual Distribution (KDE)
# 3. Feature Importance & Interpretability:
#    - Non-linear Random Forest MDI (Mean Decrease in Impurity)
#    - Linear Ridge Regularized Standardized Coefficients

print("\n" + "=" * 78)
print("PHASE 4: REGRESSION TRACK - PART E: VISUAL DIAGNOSTICS (Point 5)")
print("=" * 78)

# Extract best model predictions and residuals
best_model_name = "Tuned Random Forest"
y_best_pred = rf_tuned_preds
residuals = y_test_bike - y_best_pred

# --- Diagnostic 1: Predicted vs. Actual ---
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test_bike, y_best_pred, alpha=0.35, color='#1f77b4', edgecolors='none', s=25)

# Reference identity line (ideal y = x)
min_val = min(y_test_bike.min(), y_best_pred.min())
max_val = max(y_test_bike.max(), y_best_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Perfect Prediction (y = x)')

ax.set_title(f'Predicted vs. Actual Bike Rentals ({best_model_name})\nTest R² = {rf_tuned_res["Test R²"]:.4f} | RMSE = {rf_tuned_res["RMSE"]:.2f}', fontsize=13, pad=12)
ax.set_xlabel('Actual Rented Bike Count', fontsize=11)
ax.set_ylabel('Predicted Rented Bike Count', fontsize=11)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Diagnostic 2: Residual Diagnostics (Fitted vs Residuals & Error Distribution) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Subplot 2A: Residuals vs Fitted
axes[0].scatter(y_best_pred, residuals, alpha=0.35, color='#ff7f0e', edgecolors='none', s=25)
axes[0].axhline(0, color='red', linestyle='--', lw=2)
axes[0].set_title(f'Residuals vs. Fitted Values ({best_model_name})', fontsize=12)
axes[0].set_xlabel('Fitted / Predicted Values', fontsize=11)
axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Subplot 2B: Residual Error Distribution
sns.histplot(residuals, kde=True, ax=axes[1], color='#2ca02c', bins=40)
axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error Reference')
axes[1].axvline(residuals.mean(), color='black', linestyle=':', label=f'Mean Error: {residuals.mean():.2f}')
axes[1].set_title(f'Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})', fontsize=12)
axes[1].set_xlabel('Residual Error', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# --- Diagnostic 3: Feature Importance Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 3A: Random Forest Feature Importances
rf_importances = pd.Series(rf_search.best_estimator_.feature_importances_, index=all_feature_names_bike).sort_values(ascending=True)
rf_importances.plot(kind='barh', ax=axes[0], color='#1f77b4', edgecolor='black')
axes[0].set_title('Feature Importance: Random Forest (MDI / Gini Importance)', fontsize=12, pad=10)
axes[0].set_xlabel('Relative Importance Score', fontsize=11)
axes[0].set_ylabel('Features', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.6)

# 3B: Ridge Regression Standardized Coefficients
ridge_coefs = pd.Series(ridge_cv.best_estimator_.coef_, index=all_feature_names_bike).sort_values(ascending=True)
colors = ['#d62728' if x < 0 else '#2ca02c' for x in ridge_coefs.values]
ridge_coefs.plot(kind='barh', ax=axes[1], color=colors, edgecolor='black')
axes[1].axvline(0, color='black', lw=1.2)
axes[1].set_title(f'Feature Weights: Ridge Regression (alpha = {ridge_cv.best_params_["alpha"]})', fontsize=12, pad=10)
axes[1].set_xlabel('Standardized Coefficient Weight (Beta)', fontsize=11)
axes[1].set_ylabel('Features', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("Phase 4 Visual Diagnostics successfully completed.")
# ==============================================================================


