# Review 1 Progress Tracker

Use this document to coordinate hand-offs. Check the box `[x]` when a task is pushed to the main branch.

## Phase 1 & 2: Infrastructure & EDA
- [x] Initialize repository structure and `requirements.txt` (@Ritvik)
- [x] Complete Regression EDA & visual documentation (Seoul Bike Sharing)
- [x] Complete Classification EDA & visual documentation (Bank Marketing)

## Phase 3: Preprocessing (@Abhisarg)
- [ ] Implement missing value strategies and outlier handling for both datasets 
- [ ] Execute stratified train/test splits (random_state=42)
- [ ] Engineer at least one new feature per dataset with justification
- [ ] Apply categorical encoders and scalers (fitted on training data only)

## Phase 4: Regression Track (Seoul Bike)
- [ ] Train Linear, Ridge, Lasso, ElasticNet, and Polynomial Regression (@Abhisarg)
- [ ] Train Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN (@Ritvik)
- [ ] Hyperparameter tune at least 2 models (GridSearchCV/RandomizedSearchCV)
- [ ] Generate summary table ranking all 10 models by R² score (include RMSE, MAE)
- [ ] Plot predicted-vs-actual, residuals, and feature importance

## Phase 5: Classification Track Part A (Bank Marketing)
- [ ] Train Logistic Regression, KNN, Naive Bayes (Gaussian), Decision Tree, and SVC (@Rohan)
- [ ] Generate Accuracy, weighted F1-score, and confusion matrices for all 5 models
- [ ] Build preliminary comparison table

## Phase 6: Code Quality & Submission
- [ ] Run notebooks top-to-bottom to confirm zero execution errors
- [ ] Push final Review 1 commits to GitHub repository
