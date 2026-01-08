# Smart Study Planner - Machine Learning Project

## 📚 Project Overview

**Smart Study Planner** is an enterprise-level machine learning application that helps university students organize their study schedules efficiently. The system uses advanced ensemble learning techniques to predict required study hours for each course based on multiple factors.

**Authors:**
- Seif Ahmed Mohamed Abdou Ali Elredeini (220504545)
- Nour Mohamed Yakout Mohamed Elshabeny (240504555)

**Institution:** Istanbul Atlas University, Software Engineering Department

---

## 🎯 Key Features

- ✅ **Complete End-to-End ML Pipeline** (Chapter 2, HOML)
- ✅ **All 6 Required Ensemble Methods** (Voting, Bagging, RF, GB, AdaBoost, Stacking)
- ✅ **Real Hyperparameter Tuning** (GridSearchCV/RandomizedSearchCV)
- ✅ **Comprehensive Evaluation** (RMSE, MAE, R², MAPE, Cross-Validation)
- ✅ **Model Interpretability** (SHAP values, Feature Importance)
- ✅ **Professional Web Application** (Streamlit UI)
- ✅ **Production-Ready Code** (Enterprise-level quality)

---

## 📁 Project Structure

```
Project/
├── data/
│   ├── Simulated_DS.ipynb              # Dataset generation
│   └── smart_study_planner_dataset.csv # Dataset (1000 samples)
├── models/                              # Generated after training
│   ├── final_voting_regressor.pkl      # Trained model
│   ├── feature_scaler.pkl              # Feature scaler
│   └── *.png                           # Visualizations
├── notebooks/
│   └── 03_eda.ipynb                    # Exploratory Data Analysis
├── src/
│   ├── data_preparation.py              # Data preprocessing
│   ├── train_models.py                 # Complete model training
│   ├── final_model_analysis.py         # Model analysis
│   ├── hyperparameter_tuning.py        # Hyperparameter tuning
│   ├── comprehensive_evaluation.py     # Evaluation metrics
│   ├── model_interpretability.py       # SHAP analysis
│   └── main.py                         # Main pipeline
├── app.py                              # Web application
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
cd src
python main.py
```

This will:
- Prepare data (train/validation/test split)
- Train baseline models
- Train all 6 ensemble methods with hyperparameter tuning
- Evaluate and select best model
- Generate visualizations
- Save trained models

**Time:** 10-30 minutes (depending on hyperparameter tuning)

### 3. Run Web Application

```bash
streamlit run app.py
```

Or use convenience scripts:
- **Linux/Mac:** `./run_app.sh`
- **Windows:** `run_app.bat`

The app opens at `http://localhost:8501`

---

## 📊 Dataset

**File:** `data/smart_study_planner_dataset.csv`

- **Samples:** 1000
- **Features:** 6
- **Target:** `required_study_hours` (continuous)

**Features:**
1. `course_difficulty` (1-5): Difficulty level (1=Easy, 5=Very Difficult)
2. `credit_hours` (2-6): Number of credit hours for the course
3. `number_of_courses` (3-8): Total number of courses student is taking
4. `previous_grade` (50-100): Previous academic grade percentage
5. `days_until_exam` (1-120): Days remaining until the exam
6. `daily_available_hours` (1-6): Available study hours per day

**Target Variable:**
- `required_study_hours`: Weekly study hours required (continuous, predicted value)

---

## 🤖 Machine Learning Models

### Baseline Models
- Linear Regression
- Decision Tree Regressor

### Ensemble Methods (All 6 Required)
1. **Voting Regressor** (Hard Voting)
2. **Bagging Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **AdaBoost Regressor**
6. **Stacking Regressor**

### Hyperparameter Tuning
- **GridSearchCV** for Random Forest and AdaBoost
- **RandomizedSearchCV** for Gradient Boosting and Bagging
- 5-fold cross-validation

### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)
- Cross-Validation Scores

---

## 🎨 Web Application Features

- **Easy Course Input:** Sliders and text fields
- **Real-Time Predictions:** Instant ML-based predictions
- **Interactive Visualizations:** Plotly charts
- **Weekly & Daily Views:** Complete schedule breakdown
- **Export Options:** CSV download
- **Professional UI:** Modern, responsive design

---

## 📈 Model Performance

The final model (Voting Regressor) combines multiple ensemble methods for optimal performance:

### Best Model: Stacking Regressor
- **Validation RMSE:** 1.44 hours
- **Validation MAE:** 1.06 hours
- **Validation R²:** 0.938 (93.8% variance explained)

### Voting Regressor (Final Saved Model)
- **Validation RMSE:** 1.51 hours
- **Validation MAE:** 1.07 hours
- **Validation R²:** 0.933 (93.3% variance explained)

### All Ensemble Models Performance
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Stacking Regressor | 1.06 | 1.44 | 0.938 |
| Voting Regressor | 1.07 | 1.51 | 0.933 |
| Gradient Boosting | 1.14 | 1.57 | 0.927 |
| AdaBoost | 1.23 | 1.61 | 0.923 |
| Random Forest | 1.19 | 1.67 | 0.917 |
| Bagging | 1.19 | 1.69 | 0.916 |

*Results from validation set with hyperparameter tuning. Cross-validation: 5-fold CV.*

---

## 🔧 Technical Details

- **Problem Type:** Supervised Regression
- **Train/Val/Test Split:** 60/20/20
- **Feature Scaling:** StandardScaler
- **Cross-Validation:** 5-fold
- **Random State:** 42 (for reproducibility)

---

## 📝 Usage Examples

### Complete Pipeline (Recommended)

```bash
cd src
python main.py
```

This runs the complete pipeline: data preparation → training → evaluation → analysis.

### Running Individual Components

```bash
cd src

# Data preparation (train/val/test split + scaling)
python data_preparation.py

# Model training (baseline + all 6 ensemble methods)
python train_models.py

# Final model analysis (visualizations + metrics)
python final_model_analysis.py
```

**Note:** Run `main.py` for the complete pipeline. Individual scripts are for development/debugging.

### Using Jupyter Notebooks

```bash
# Generate dataset
jupyter notebook data/Simulated_DS.ipynb

# Exploratory Data Analysis
jupyter notebook notebooks/03_eda.ipynb
```

---

## 📦 Dependencies

See `requirements.txt` for complete list. Key packages:

- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.4.0`
- `plotly>=5.0.0`
- `streamlit>=1.28.0`
- `joblib>=1.0.0`
- `shap>=0.40.0` (for interpretability)
- `scipy>=1.7.0`
- `seaborn>=0.11.0`

---

## 📊 Output Files

After running the pipeline, generated in `models/`:

### Models
- `final_voting_regressor.pkl` - Trained final model (for production)
- `feature_scaler.pkl` - Feature scaler (required for predictions)
- `all_ensemble_models.pkl` - All trained ensemble models

### Results & Analysis
- `model_comparison.csv` - Performance comparison of all models
- `feature_importance.png` - Feature importance visualization
- `prediction_vs_actual.png` - Prediction vs actual scatter plot

### Additional Files (if comprehensive evaluation is run)
- `*_metrics.csv` - Detailed metrics per model
- `*_residuals.png` - Residual analysis plots
- `*_learning_curves.png` - Learning curve analysis
- `*_error_distribution.png` - Error distribution plots

---

## 🎓 Academic Compliance

### Chapter 2 Pipeline (HOML)
✅ All 8 steps implemented:
1. Frame the problem
2. Get the data
3. Explore the data
4. Prepare the data
5. Select and train models
6. Fine-tune models
7. Present solution
8. Launch system

### Ensemble Methods
✅ All 6 required methods implemented and evaluated

### Evaluation
✅ Comprehensive metrics with cross-validation

---

## 🐛 Troubleshooting

### Model files not found
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'models/final_voting_regressor.pkl'`

**Solution:**
```bash
# Run training pipeline first
cd src
python main.py
```

### Import errors
**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Port already in use (Streamlit)
**Error:** `Port 8501 is already in use`

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Dataset not found
**Error:** `FileNotFoundError: ../data/smart_study_planner_dataset.csv`

**Solution:**
```bash
# Generate dataset first
jupyter notebook data/Simulated_DS.ipynb
# Run all cells to generate the CSV file
```

---

## 📄 License

This project is developed for academic purposes as part of the Machine Learning course at Istanbul Atlas University.

---

## 👥 Contact

For questions or issues:
- Seif Ahmed Mohamed Abdou Ali Elredeini
- Nour Mohamed Yakout Mohamed Elshabeny

---

## 📌 Quick Reference

### Essential Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Train models (first time)
cd src && python main.py

# Run web app
streamlit run app.py
```

### File Locations
- **Dataset:** `data/smart_study_planner_dataset.csv`
- **Trained Models:** `models/`
- **Source Code:** `src/`
- **Web App:** `app.py`

---

**Built with ❤️ using Python and Scikit-learn**

*Last Updated: 2024*
