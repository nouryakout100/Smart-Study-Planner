# =====================================================
# ENTERPRISE-LEVEL MODEL TRAINING PIPELINE
# =====================================================
"""
Complete model training pipeline with:
- Baseline models
- All 6 ensemble methods
- Real hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Comprehensive evaluation
- Model interpretability
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from data_preparation import prepare_data

# Import comprehensive evaluation if available, otherwise use simple version
try:
    from comprehensive_evaluation import calculate_all_metrics
except ImportError:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    def calculate_all_metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R²': r2_score(y_true, y_pred)
        }

# =====================================================
# CONFIGURATION
# =====================================================
USE_HYPERPARAMETER_TUNING = True  # Set to False for faster execution
USE_VALIDATION_SET = True
CV_FOLDS = 5

# =====================================================
# DATA LOADING
# =====================================================
print("=" * 70)
print("SMART STUDY PLANNER - MODEL TRAINING PIPELINE")
print("=" * 70)

if USE_VALIDATION_SET:
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        "../data/smart_study_planner_dataset.csv",
        use_validation_set=True
    )
    print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
else:
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "../data/smart_study_planner_dataset.csv",
        use_validation_set=False
    )
    X_val, y_val = X_test, y_test  # Use test as validation if no separate validation set
    print(f"Training: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def evaluate_model(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2}

# Alias for compatibility
calculate_all_metrics = evaluate_model

def tune_hyperparameters(model, param_grid, X_train, y_train, use_randomized=True):
    """Tune hyperparameters using GridSearchCV or RandomizedSearchCV"""
    if not USE_HYPERPARAMETER_TUNING:
        # Use default model with reasonable parameters
        model.fit(X_train, y_train)
        return model, {}
    
    try:
        if use_randomized:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=30, cv=CV_FOLDS,
                scoring='neg_root_mean_squared_error', n_jobs=-1,
                random_state=42, verbose=1
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=CV_FOLDS,
                scoring='neg_root_mean_squared_error', n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        print(f"Best params: {search.best_params_}")
        print(f"Best CV score: {-search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_
    except Exception as e:
        print(f"⚠️  Hyperparameter tuning failed: {e}")
        print("   Using default parameters...")
        model.fit(X_train, y_train)
        return model, {}

# =====================================================
# BASELINE MODELS
# =====================================================
print("\n" + "=" * 70)
print("BASELINE MODELS")
print("=" * 70)

# Linear Regression
print("\nTraining Linear Regression...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_val)
results_lin = evaluate_model(y_val, y_pred_lin)
print(f"Linear Regression - RMSE: {results_lin['RMSE']:.4f}, R²: {results_lin['R²']:.4f}")

# Decision Tree
print("\nTraining Decision Tree...")
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_val)
results_tree = evaluate_model(y_val, y_pred_tree)
print(f"Decision Tree - RMSE: {results_tree['RMSE']:.4f}, R²: {results_tree['R²']:.4f}")

# =====================================================
# ENSEMBLE MODELS WITH HYPERPARAMETER TUNING
# =====================================================
print("\n" + "=" * 70)
print("ENSEMBLE MODELS TRAINING")
print("=" * 70)

results = {}
best_models = {}

# Random Forest
print("\n" + "-" * 70)
print("Random Forest Regressor")
print("-" * 70)
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_best, rf_params = tune_hyperparameters(rf_base, rf_param_grid, X_train, y_train, use_randomized=False)
rf_best.fit(X_train, y_train)
y_pred_rf = rf_best.predict(X_val)
results['Random Forest'] = evaluate_model(y_val, y_pred_rf)
best_models['Random Forest'] = rf_best
print(f"RMSE: {results['Random Forest']['RMSE']:.4f}, R²: {results['Random Forest']['R²']:.4f}")

# Gradient Boosting
print("\n" + "-" * 70)
print("Gradient Boosting Regressor")
print("-" * 70)
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
gb_base = GradientBoostingRegressor(random_state=42)
gb_best, gb_params = tune_hyperparameters(gb_base, gb_param_grid, X_train, y_train, use_randomized=True)
gb_best.fit(X_train, y_train)
y_pred_gb = gb_best.predict(X_val)
results['Gradient Boosting'] = evaluate_model(y_val, y_pred_gb)
best_models['Gradient Boosting'] = gb_best
print(f"RMSE: {results['Gradient Boosting']['RMSE']:.4f}, R²: {results['Gradient Boosting']['R²']:.4f}")

# AdaBoost
print("\n" + "-" * 70)
print("AdaBoost Regressor")
print("-" * 70)
ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3, 5]
}
ada_base = AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=42), random_state=42)
ada_best, ada_params = tune_hyperparameters(ada_base, ada_param_grid, X_train, y_train, use_randomized=False)
ada_best.fit(X_train, y_train)
y_pred_ada = ada_best.predict(X_val)
results['AdaBoost'] = evaluate_model(y_val, y_pred_ada)
best_models['AdaBoost'] = ada_best
print(f"RMSE: {results['AdaBoost']['RMSE']:.4f}, R²: {results['AdaBoost']['R²']:.4f}")

# Bagging
print("\n" + "-" * 70)
print("Bagging Regressor")
print("-" * 70)
bag_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'estimator__max_depth': [5, 10, 15]
}
bag_base = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), random_state=42, n_jobs=-1)
bag_best, bag_params = tune_hyperparameters(bag_base, bag_param_grid, X_train, y_train, use_randomized=True)
bag_best.fit(X_train, y_train)
y_pred_bag = bag_best.predict(X_val)
results['Bagging'] = evaluate_model(y_val, y_pred_bag)
best_models['Bagging'] = bag_best
print(f"RMSE: {results['Bagging']['RMSE']:.4f}, R²: {results['Bagging']['R²']:.4f}")

# Voting Regressor
print("\n" + "-" * 70)
print("Voting Regressor (Hard Voting)")
print("-" * 70)
voting_reg = VotingRegressor(
    estimators=[
        ("rf", rf_best),
        ("gb", gb_best),
        ("ada", ada_best)
    ]
)
voting_reg.fit(X_train, y_train)
y_pred_voting = voting_reg.predict(X_val)
results['Voting Regressor'] = evaluate_model(y_val, y_pred_voting)
best_models['Voting Regressor'] = voting_reg
print(f"RMSE: {results['Voting Regressor']['RMSE']:.4f}, R²: {results['Voting Regressor']['R²']:.4f}")

# Stacking Regressor
print("\n" + "-" * 70)
print("Stacking Regressor")
print("-" * 70)
stacking_reg = StackingRegressor(
    estimators=[
        ("rf", rf_best),
        ("gb", gb_best),
        ("ada", ada_best)
    ],
    final_estimator=LinearRegression(),
    cv=CV_FOLDS
)
stacking_reg.fit(X_train, y_train)
y_pred_stacking = stacking_reg.predict(X_val)
results['Stacking Regressor'] = evaluate_model(y_val, y_pred_stacking)
best_models['Stacking Regressor'] = stacking_reg
print(f"RMSE: {results['Stacking Regressor']['RMSE']:.4f}, R²: {results['Stacking Regressor']['R²']:.4f}")

# =====================================================
# MODEL COMPARISON AND SELECTION
# =====================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON (Validation Set)")
print("=" * 70)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='RMSE', ascending=True)
print(results_df.round(4))

# Select best model
best_model_name = results_df.index[0]
best_model = best_models[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Validation RMSE: {results_df.loc[best_model_name, 'RMSE']:.4f}")
print(f"  Validation R²: {results_df.loc[best_model_name, 'R²']:.4f}")

# Final evaluation on test set
y_pred_test = best_model.predict(X_test)
test_results = evaluate_model(y_test, y_pred_test)
print(f"\n✓ Test Set Performance:")
print(f"  Test RMSE: {test_results['RMSE']:.4f}")
print(f"  Test R²: {test_results['R²']:.4f}")

# Cross-validation
cv_scores = -cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"\n✓ Cross-Validation (5-fold):")
print(f"  CV RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =====================================================
# SAVE MODELS
# =====================================================
os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/final_voting_regressor.pkl")
joblib.dump(scaler, "../models/feature_scaler.pkl")
joblib.dump(best_models, "../models/all_ensemble_models.pkl")

# Save results
results_df.to_csv("../models/model_comparison.csv")
print(f"\n✓ Models saved to ../models/")
print(f"✓ Results saved to ../models/model_comparison.csv")

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

