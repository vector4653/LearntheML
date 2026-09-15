# Review 1 Progress Tracker

Use this document to coordinate hand-offs. Check the box `[x]` when a task is pushed to the main branch.

## Phase 1 & 2: Infrastructure & EDA
- [x] Initialize repository structure and `requirements.txt` (@Ritvik)
- [x] Complete Regression EDA & visual documentation (Seoul Bike Sharing)
- [x] Complete Classification EDA & visual documentation (Bank Marketing)

## Phase 3: Preprocessing (@Abhisarg)
- [x] Implement missing value strategies and outlier handling for both datasets 
- [x] Execute stratified train/test splits (random_state=42)
- [x] Engineer at least one new feature per dataset with justification
- [x] Apply categorical encoders and scalers (fitted on training data only)

## Phase 4: Regression Track (Seoul Bike)
- [x] Train Linear, Ridge, Lasso, ElasticNet, and Polynomial Regression (@Abhisarg)
- [x] Train Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN (@Ritvik)
- [x] Hyperparameter tune at least 2 models (GridSearchCV/RandomizedSearchCV)
- [x] Generate summary table ranking all 10 models by R² score (include RMSE, MAE)
- [x] Plot predicted-vs-actual, residuals, and feature importance

## Phase 5: Classification Track Part A (Bank Marketing)
- [x] Train Logistic Regression, KNN, Naive Bayes (Gaussian), Decision Tree, and SVC (@Rohan)
- [x] Generate Accuracy, weighted F1-score, and confusion matrices for all 5 models
- [x] Build preliminary comparison table

## Phase 6: Code Quality & Submission
- [ ] Run notebooks top-to-bottom to confirm zero execution errors
- [ ] Push final Review 1 commits to GitHub repository
