# =====================================================
# COMPREHENSIVE MODEL EVALUATION
# =====================================================
"""
This module provides comprehensive evaluation metrics and analysis:
- R² score (coefficient of determination)
- MAPE (Mean Absolute Percentage Error)
- Residual analysis
- Learning curves
- Overfitting detection
- Error distribution analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve
import os

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (handle division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    # Mean percentage error
    mpe = np.mean((y_true - y_pred) / np.where(y_true != 0, y_true, 1)) * 100
    
    # Explained variance score
    explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R²': r2,
        'MAPE (%)': mape,
        'MPE (%)': mpe,
        'Explained Variance': explained_variance
    }
    
    return metrics


def plot_residuals(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create comprehensive residual analysis plots.
    
    Plots:
    1. Residuals vs Predicted values
    2. Residuals distribution (histogram)
    3. Q-Q plot for normality check
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title(f'Residuals vs Predicted - {model_name}', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Residuals distribution
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Residuals Distribution - {model_name}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Q-Q plot for normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f'Q-Q Plot (Normality Check) - {model_name}', fontsize=14, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual analysis plot saved to {save_path}")
    
    plt.close()


def plot_learning_curves(model, X_train, y_train, cv=5, train_sizes=None, save_path=None):
    """
    Plot learning curves to detect overfitting/underfitting.
    
    Parameters:
    -----------
    model : estimator
        The model to evaluate
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    cv : int, default=5
        Cross-validation folds
    train_sizes : array-like, optional
        Training set sizes to evaluate
    save_path : str, optional
        Path to save the plot
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv,
        train_sizes=train_sizes,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    # Convert to positive RMSE
    train_scores = -train_scores
    val_scores = -val_scores
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Learning Curves (Overfitting Detection)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    plt.close()
    
    # Check for overfitting
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    if gap > 0.5:  # Threshold for overfitting
        print(f"⚠️  WARNING: Potential overfitting detected!")
        print(f"   Training RMSE: {final_train_score:.4f}")
        print(f"   Validation RMSE: {final_val_score:.4f}")
        print(f"   Gap: {gap:.4f}")
    else:
        print(f"✓ No significant overfitting detected")
        print(f"   Training RMSE: {final_train_score:.4f}")
        print(f"   Validation RMSE: {final_val_score:.4f}")
        print(f"   Gap: {gap:.4f}")


def plot_error_distribution(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot error distribution and statistics.
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Error distribution
    axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Error Distribution - {model_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Absolute error distribution
    axes[1].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[1].axvline(x=np.mean(abs_errors), color='g', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(abs_errors):.2f}')
    axes[1].axvline(x=np.median(abs_errors), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(abs_errors):.2f}')
    axes[1].set_xlabel('Absolute Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Absolute Error Distribution - {model_name}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    
    plt.close()
    
    # Print statistics
    print(f"\nError Statistics for {model_name}:")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Std Error: {np.std(errors):.4f}")
    print(f"  Mean Absolute Error: {np.mean(abs_errors):.4f}")
    print(f"  Median Absolute Error: {np.median(abs_errors):.4f}")
    print(f"  Max Error: {np.max(abs_errors):.4f}")
    print(f"  Min Error: {np.min(abs_errors):.4f}")


def comprehensive_evaluation(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                             model_name="Model", save_dir="../models"):
    """
    Perform comprehensive evaluation of a model.
    
    Returns:
    --------
    results : dict
        Dictionary containing all evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print("=" * 70)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for all sets
    metrics_train = calculate_all_metrics(y_train, y_pred_train)
    metrics_val = calculate_all_metrics(y_val, y_pred_val)
    metrics_test = calculate_all_metrics(y_test, y_pred_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Training': metrics_train,
        'Validation': metrics_val,
        'Test': metrics_test
    }).T
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE METRICS")
    print("=" * 70)
    print(results_df.round(4))
    
    # Save metrics
    results_df.to_csv(f"{save_dir}/{model_name.lower().replace(' ', '_')}_metrics.csv")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING ANALYSIS PLOTS")
    print("=" * 70)
    
    # Residual analysis (on test set)
    plot_residuals(
        y_test, y_pred_test, 
        model_name=model_name,
        save_path=f"{save_dir}/{model_name.lower().replace(' ', '_')}_residuals.png"
    )
    
    # Learning curves
    plot_learning_curves(
        model, X_train, y_train,
        cv=5,
        save_path=f"{save_dir}/{model_name.lower().replace(' ', '_')}_learning_curves.png"
    )
    
    # Error distribution
    plot_error_distribution(
        y_test, y_pred_test,
        model_name=model_name,
        save_path=f"{save_dir}/{model_name.lower().replace(' ', '_')}_error_distribution.png"
    )
    
    results = {
        'metrics': results_df,
        'predictions': {
            'train': y_pred_train,
            'val': y_pred_val,
            'test': y_pred_test
        }
    }
    
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {model_name}")
    print("=" * 70)
    
    return results

