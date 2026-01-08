# =====================================================
# DATA PREPARATION FOR MACHINE LEARNING ALGORITHMS
# =====================================================
"""
This module implements Step 5 of the End-to-End ML Pipeline (Chapter 2, HOML):
"Prepare the Data for Machine Learning Algorithms"

Following the methodology from Chapter 2:
- Load the dataset
- Separate features and target
- Create train-test split (20% test set, as recommended in PDF)
- Apply feature scaling (StandardScaler)

Note: According to Chapter 2, the test set should be created early and set aside.
In this implementation, we create it during data preparation before model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(csv_path):
    return pd.read_csv(csv_path)


## Step 4.2 — Separate Features and Target

def split_features_target(df):
    X = df.drop("required_study_hours", axis=1)
    y = df["required_study_hours"]
    return X, y


## Step 4.3 — Train / Test Split
## Following Chapter 2: "Pick some instances randomly, typically 20% of the dataset"
def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Following Chapter 2 methodology:
    - 20% of dataset for testing (as recommended in PDF)
    - Random state=42 for reproducibility
    - Test set is set aside and not used until final evaluation
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

## Step 4.4 — Feature Scaling
## Following Chapter 2: StandardScaler for feature normalization
def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Following Chapter 2 methodology:
    - Fit scaler on training data only
    - Transform both training and test data
    - Prevents data leakage (test set never influences scaling)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


## Step 4.5 — Train/Validation/Test Split (Improved)
def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Following best practices:
    - 60% training (for model training)
    - 20% validation (for hyperparameter tuning)
    - 20% test (for final evaluation only)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation
    # Adjust val_size relative to temp set
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


## Step 4.6 — One Function to Rule Them All
def prepare_data(csv_path, use_validation_set=True):
    """
    Prepare data for machine learning.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    use_validation_set : bool, default=True
        If True, returns train/val/test split
        If False, returns train/test split (for backward compatibility)
    """
    df = load_data(csv_path)
    X, y = split_features_target(df)
    
    if use_validation_set:
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features_with_val(
            X_train, X_val, X_test
        )
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
    else:
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def scale_features_with_val(X_train, X_val, X_test):
    """
    Scale features for train, validation, and test sets.
    Fit scaler only on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


## Step 4.7 — Quick Sanity Check
if __name__ == "__main__":
    # Test with validation set
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(
        "../data/smart_study_planner_dataset.csv",
        use_validation_set=True
    )
    print("Training set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Test set shape:", X_test.shape)
    
    # Test without validation set (backward compatibility)
    X_train_old, X_test_old, y_train_old, y_test_old, scaler_old = prepare_data(
        "../data/smart_study_planner_dataset.csv",
        use_validation_set=False
    )
    print("\nBackward compatibility test:")
    print("Training set shape:", X_train_old.shape)
    print("Test set shape:", X_test_old.shape)



