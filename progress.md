# Project Progress Tracker

Use this document to coordinate hand-offs. Check the box `[x]` when a task is pushed to the main branch.

## Review 1 Milestones

### Phase 1 & 2: Infrastructure & EDA
- [x] Initialize repository structure and `requirements.txt` (@Ritvik)
- [x] Complete Regression EDA & visual documentation (Seoul Bike Sharing)
- [x] Complete Classification EDA & visual documentation (Bank Marketing)

### Phase 3: Preprocessing (@Abhisarg)
- [x] Implement missing value strategies and outlier handling for both datasets 
- [x] Execute stratified train/test splits (random_state=42)
- [x] Engineer at least one new feature per dataset with justification
- [x] Apply categorical encoders and scalers (fitted on training data only)

### Phase 4: Regression Track (Seoul Bike)
- [x] Train Linear, Ridge, Lasso, ElasticNet, and Polynomial Regression (@Abhisarg)
- [x] Train Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN (@Ritvik)
- [x] Hyperparameter tune at least 2 models (GridSearchCV/RandomizedSearchCV)
- [x] Generate summary table ranking all 10 models by R² score (include RMSE, MAE)
- [x] Plot predicted-vs-actual, residuals, and feature importance

### Phase 5: Classification Track Part A (Bank Marketing)
- [x] Train Logistic Regression, KNN, Naive Bayes (Gaussian), Decision Tree, and SVC (@Rohan)
- [x] Generate Accuracy, weighted F1-score, and confusion matrices for all 5 models
- [x] Build preliminary comparison table

### Phase 6: Code Quality & Submission
- [x] Run notebooks top-to-bottom to confirm zero execution errors
- [x] Push final Review 1 commits to GitHub repository

---

## Review 2 Milestones

### Section B: Clustering Track (Spotify Audio Profiles)
- [x] Data audit, deduplication, and feature selection for Spotify 114k tracks (`notebooks/clustering.ipynb`)
- [x] Isotropic feature standardization (`StandardScaler`)
- [x] Systematic hyperparameter search ($k \in [2, 8]$) with Elbow curve (Inertia/WCSS), Silhouette, Davies-Bouldin, and Calinski-Harabasz curves
- [x] Fit K-Means ($k=5$) with `k-means++` and extract cluster centroids (Rubric B1)
- [x] Compute Ward hierarchical linkage matrix and plot truncated Dendrogram with $k=5$ threshold cut (Rubric B1, B2)
- [x] Master clustering evaluation table reporting Silhouette, Davies-Bouldin, Calinski-Harabasz, and cross-algorithm ARI/NMI (Rubric B2)
- [x] 2D PCA cluster projection for both algorithms with K-Means centroids (Rubric B3)
- [x] 2D t-SNE non-linear manifold projection (Rubric B3)
- [x] Audio archetype persona profiling and music recommendation takeaways (Rubric D1)
- [x] Script-first modular workflow implemented (`notebooks/py_files/clustering.py`)
