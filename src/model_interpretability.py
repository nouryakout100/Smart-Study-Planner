# =====================================================
# MODEL INTERPRETABILITY WITH SHAP VALUES
# =====================================================
"""
This module provides model interpretability using SHAP (SHapley Additive exPlanations)
values to understand model predictions and feature contributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Install with: pip install shap")

def explain_model_with_shap(model, X_train, X_test, feature_names, model_name="Model", save_dir="../models"):
    """
    Generate SHAP explanations for model predictions.
    
    Parameters:
    -----------
    model : estimator
        Trained model (must be tree-based for TreeExplainer)
    X_train : array-like
        Training data (for background)
    X_test : array-like
        Test data to explain
    feature_names : list
        Names of features
    model_name : str
        Name of the model
    save_dir : str
        Directory to save plots
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping SHAP analysis.")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"SHAP ANALYSIS: {model_name}")
    print("=" * 70)
    
    # Create SHAP explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except:
        # Fallback to KernelExplainer for non-tree models
        print("Using KernelExplainer (slower but works for all models)...")
        explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Sample for speed
        shap_values = explainer.shap_values(X_test[:50])  # Sample for speed
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP summary plot saved")
    
    # Bar plot of mean SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_shap_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ SHAP bar plot saved")
    
    # Calculate feature importance from SHAP
    if len(shap_values.shape) == 2:
        mean_shap_values = np.abs(shap_values).mean(0)
    else:
        mean_shap_values = np.abs(shap_values).mean()
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP Value|': mean_shap_values
    }).sort_values(by='Mean |SHAP Value|', ascending=False)
    
    print("\nFeature Importance (from SHAP):")
    print(feature_importance_df.to_string(index=False))
    
    # Save feature importance
    feature_importance_df.to_csv(f"{save_dir}/{model_name.lower().replace(' ', '_')}_shap_importance.csv", index=False)
    
    return feature_importance_df


def plot_partial_dependence(model, X_train, feature_names, feature_idx, save_path=None):
    """
    Plot partial dependence for a specific feature.
    """
    from sklearn.inspection import PartialDependenceDisplay
    
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, [feature_idx], feature_names=feature_names
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Partial dependence plot saved to {save_path}")
    
    plt.close()


def comprehensive_interpretability(model, X_train, X_test, feature_names, model_name="Model", save_dir="../models"):
    """
    Comprehensive model interpretability analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"COMPREHENSIVE INTERPRETABILITY: {model_name}")
    print("=" * 70)
    
    # SHAP analysis
    shap_importance = explain_model_with_shap(
        model, X_train, X_test, feature_names, model_name, save_dir
    )
    
    # Partial dependence plots for top features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features_idx = np.argsort(importances)[-3:][::-1]  # Top 3 features
        
        print("\nGenerating partial dependence plots for top 3 features...")
        for i, idx in enumerate(top_features_idx):
            plot_partial_dependence(
                model, X_train, feature_names, idx,
                save_path=f"{save_dir}/{model_name.lower().replace(' ', '_')}_pd_feature_{feature_names[idx]}.png"
            )
    
    print("\n" + "=" * 70)
    print(f"INTERPRETABILITY ANALYSIS COMPLETE: {model_name}")
    print("=" * 70)
    
    return shap_importance

