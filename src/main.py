# =====================================================
# SMART STUDY PLANNER - ENTERPRISE MAIN PIPELINE
# =====================================================
"""
Complete end-to-end machine learning pipeline:
1. Data Preparation
2. Baseline Model Training
3. Ensemble Model Training (with hyperparameter tuning)
4. Model Evaluation and Selection
5. Final Model Analysis
6. Study Schedule Generation

This is the main entry point for the complete pipeline.
"""

import sys
import os

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)
os.chdir(src_dir)

def main():
    """Execute complete ML pipeline"""
    print("=" * 70)
    print("SMART STUDY PLANNER - ENTERPRISE ML PIPELINE")
    print("=" * 70)
    print("\nPipeline includes:")
    print("  ✓ Data preparation with train/validation/test split")
    print("  ✓ Baseline models (Linear Regression, Decision Tree)")
    print("  ✓ All 6 ensemble methods with hyperparameter tuning")
    print("  ✓ Comprehensive evaluation (RMSE, MAE, R², MAPE)")
    print("  ✓ Model comparison and selection")
    print("  ✓ Cross-validation")
    print("=" * 70)
    
    try:
        # Step 1: Data Preparation
        print("\n" + "=" * 70)
        print("STEP 1: DATA PREPARATION")
        print("=" * 70)
        from data_preparation import prepare_data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
            "../data/smart_study_planner_dataset.csv",
            use_validation_set=True
        )
        print("✓ Data preparation completed")
        print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Step 2: Model Training
        print("\n" + "=" * 70)
        print("STEP 2: MODEL TRAINING")
        print("=" * 70)
        print("Training baseline and ensemble models...")
        import train_models
        print("✓ Model training completed")
        
        # Step 3: Final Analysis
        print("\n" + "=" * 70)
        print("STEP 3: FINAL MODEL ANALYSIS")
        print("=" * 70)
        print("Performing comprehensive analysis...")
        import final_model_analysis
        print("✓ Final analysis completed")
        
        # Success
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated outputs:")
        print("  - Final model: ../models/final_voting_regressor.pkl")
        print("  - Feature scaler: ../models/feature_scaler.pkl")
        print("  - All ensemble models: ../models/all_ensemble_models.pkl")
        print("  - Model comparison: ../models/model_comparison.csv")
        print("  - Visualizations: ../models/*.png")
        print("\n" + "=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
