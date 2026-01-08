# =====================================================
# HYPERPARAMETER TUNING WITH GRID SEARCH
# =====================================================
"""
This module implements proper hyperparameter tuning using GridSearchCV
and RandomizedSearchCV for all ensemble models.

This addresses the critical issue of claiming hyperparameter tuning
without actually performing it.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import joblib
import os

from data_preparation import prepare_data

# Custom RMSE scorer
rmse_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False
)

def tune_random_forest(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """Tune Random Forest hyperparameters using GridSearchCV"""
    print("=" * 70)
    print("HYPERPARAMETER TUNING: Random Forest Regressor")
    print("=" * 70)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {-grid_search.best_score_:.4f}")
    print(f"Best estimator: {grid_search.best_estimator_}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def tune_gradient_boosting(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """Tune Gradient Boosting hyperparameters using RandomizedSearchCV"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: Gradient Boosting Regressor")
    print("=" * 70)
    
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    
    random_search = RandomizedSearchCV(
        gb,
        param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42
    )
    
    print("Performing randomized search (50 iterations)...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score (RMSE): {-random_search.best_score_:.4f}")
    print(f"Best estimator: {random_search.best_estimator_}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_adaboost(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """Tune AdaBoost hyperparameters using GridSearchCV"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: AdaBoost Regressor")
    print("=" * 70)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'estimator__max_depth': [1, 2, 3, 5]
    }
    
    base_estimator = DecisionTreeRegressor(random_state=42)
    ada = AdaBoostRegressor(estimator=base_estimator, random_state=42)
    
    grid_search = GridSearchCV(
        ada,
        param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {-grid_search.best_score_:.4f}")
    print(f"Best estimator: {grid_search.best_estimator_}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def tune_bagging(X_train, y_train, cv=5, n_jobs=-1, verbose=1):
    """Tune Bagging Regressor hyperparameters using RandomizedSearchCV"""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING: Bagging Regressor")
    print("=" * 70)
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_samples': [0.5, 0.7, 0.9, 1.0],
        'max_features': [0.5, 0.7, 0.9, 1.0],
        'estimator__max_depth': [5, 10, 15, None]
    }
    
    base_estimator = DecisionTreeRegressor(random_state=42)
    bag = BaggingRegressor(estimator=base_estimator, random_state=42, n_jobs=n_jobs)
    
    random_search = RandomizedSearchCV(
        bag,
        param_distributions,
        n_iter=30,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=42
    )
    
    print("Performing randomized search (30 iterations)...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score (RMSE): {-random_search.best_score_:.4f}")
    print(f"Best estimator: {random_search.best_estimator_}")
    
    return random_search.best_estimator_, random_search.best_params_


def tune_all_models(X_train, y_train, cv=5, save_results=True):
    """
    Tune hyperparameters for all models and return best estimators.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    save_results : bool, default=True
        Whether to save tuning results to file
        
    Returns:
    --------
    best_models : dict
        Dictionary with best tuned models
    best_params : dict
        Dictionary with best parameters for each model
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Cross-validation folds: {cv}")
    print("=" * 70)
    
    best_models = {}
    best_params = {}
    
    # Tune Random Forest
    rf_best, rf_params = tune_random_forest(X_train, y_train, cv=cv)
    best_models['Random Forest'] = rf_best
    best_params['Random Forest'] = rf_params
    
    # Tune Gradient Boosting
    gb_best, gb_params = tune_gradient_boosting(X_train, y_train, cv=cv)
    best_models['Gradient Boosting'] = gb_best
    best_params['Gradient Boosting'] = gb_params
    
    # Tune AdaBoost
    ada_best, ada_params = tune_adaboost(X_train, y_train, cv=cv)
    best_models['AdaBoost'] = ada_best
    best_params['AdaBoost'] = ada_params
    
    # Tune Bagging
    bag_best, bag_params = tune_bagging(X_train, y_train, cv=cv)
    best_models['Bagging'] = bag_best
    best_params['Bagging'] = bag_params
    
    # Save results
    if save_results:
        os.makedirs("../models", exist_ok=True)
        results_df = pd.DataFrame(best_params).T
        results_df.to_csv("../models/hyperparameter_tuning_results.csv")
        print("\n✓ Hyperparameter tuning results saved to ../models/hyperparameter_tuning_results.csv")
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 70)
    
    return best_models, best_params


if __name__ == "__main__":
    # Load prepared data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "../data/smart_study_planner_dataset.csv"
    )
    
    # Perform hyperparameter tuning
    best_models, best_params = tune_all_models(X_train, y_train, cv=5)
    
    print("\n" + "=" * 70)
    print("SUMMARY OF BEST PARAMETERS")
    print("=" * 70)
    for model_name, params in best_params.items():
        print(f"\n{model_name}:")
        for param, value in params.items():
            print(f"  {param}: {value}")

