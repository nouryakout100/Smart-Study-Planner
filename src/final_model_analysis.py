# =====================================================
# FINAL MODEL ANALYSIS AND VISUALIZATION
# =====================================================
"""
Comprehensive analysis of the final model:
- Feature importance
- Prediction vs actual plots
- Model performance metrics
- Cross-validation results
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from data_preparation import prepare_data

# Feature names
FEATURE_NAMES = [
    "course_difficulty",
    "credit_hours",
    "number_of_courses",
    "previous_grade",
    "days_until_exam",
    "daily_available_hours"
]

# Load data
X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
    "../data/smart_study_planner_dataset.csv",
    use_validation_set=True
)

# Load best model
print("=" * 70)
print("FINAL MODEL ANALYSIS")
print("=" * 70)

try:
    final_model = joblib.load("../models/final_voting_regressor.pkl")
    print("✓ Loaded final model")
except FileNotFoundError:
    print("⚠️  Model not found. Please run train_models.py first.")
    exit(1)

# Predictions
y_pred_train = final_model.predict(X_train)
y_pred_val = final_model.predict(X_val)
y_pred_test = final_model.predict(X_test)

# Metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred)
    }

train_metrics = calculate_metrics(y_train, y_pred_train)
val_metrics = calculate_metrics(y_val, y_pred_val)
test_metrics = calculate_metrics(y_test, y_pred_test)

print("\nModel Performance:")
print(f"  Training - RMSE: {train_metrics['RMSE']:.4f}, R²: {train_metrics['R²']:.4f}")
print(f"  Validation - RMSE: {val_metrics['RMSE']:.4f}, R²: {val_metrics['R²']:.4f}")
print(f"  Test - RMSE: {test_metrics['RMSE']:.4f}, R²: {test_metrics['R²']:.4f}")

# Cross-validation
cv_scores = -cross_val_score(final_model, X_train, y_train, cv=5,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"\nCross-Validation (5-fold):")
print(f"  CV RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance (from Random Forest if available)
if hasattr(final_model, 'estimators_'):
    # Get feature importance from first estimator
    rf_estimator = None
    for est in final_model.estimators_:
        if hasattr(est, 'feature_importances_'):
            rf_estimator = est
            break
    
    if rf_estimator:
        importances = rf_estimator.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_df.to_string(index=False))
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(feature_df['Feature'], feature_df['Importance'], color='steelblue')
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        os.makedirs("../models", exist_ok=True)
        plt.savefig("../models/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved")

# Prediction vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Study Hours')
plt.ylabel('Predicted Study Hours')
plt.title('Predicted vs Actual Study Hours')
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("../models", exist_ok=True)
plt.savefig("../models/prediction_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Prediction vs actual plot saved")

print("\n" + "=" * 70)
print("FINAL ANALYSIS COMPLETE")
print("=" * 70)

