Smart Study Planner - Machine Learning Project
README Document

Authors:
Seif Ahmed Mohamed Abdou Ali Elredeini (220504545)
Nour Mohamed Yakout Mohamed Elshabeny (240504555)

Software Engineering Department
Istanbul Atlas University

═══════════════════════════════════════════════════════════════════════════════

1. ONLINE CODE REPOSITORY

GitHub Repository:
https://github.com/[YOUR_USERNAME]/Smart-Study-Planner

Google Colab:
https://colab.research.google.com/drive/[YOUR_COLAB_ID]

Kaggle:
https://www.kaggle.com/code/[YOUR_USERNAME]/smart-study-planner

Dataset Location:
The dataset (smart_study_planner_dataset.csv) is included in the repository under data/ folder.

═══════════════════════════════════════════════════════════════════════════════

2. VIDEO PRESENTATION

YouTube Link:
https://www.youtube.com/watch?v=[YOUR_VIDEO_ID]

═══════════════════════════════════════════════════════════════════════════════

3. GOOGLE DRIVE (For Files > 50 MB)

If project files exceed 50 MB, download from:
https://drive.google.com/drive/folders/[YOUR_FOLDER_ID]

Note: Replace [YOUR_FOLDER_ID] with your actual Google Drive folder ID if needed.

═══════════════════════════════════════════════════════════════════════════════

4. REQUIRED LIBRARY PACKAGES

Install all dependencies:

pip install -r requirements.txt

Required Packages:
• numpy>=1.21.0
• pandas>=1.3.0
• scipy>=1.7.0
• scikit-learn>=1.0.0
• matplotlib>=3.4.0
• plotly>=5.0.0
• seaborn>=0.11.0
• shap>=0.40.0
• streamlit>=1.28.0
• joblib>=1.0.0
• jupyter>=1.0.0 (optional, for notebooks)
• ipykernel>=6.0.0 (optional, for notebooks)

═══════════════════════════════════════════════════════════════════════════════

5. INSTRUCTIONS TO RUN THE PROJECT

Prerequisites:
• Python 3.11 or higher
• pip package manager

Step 1: Install Dependencies

pip install -r requirements.txt

Step 2: Generate Dataset (If Not Already Generated)

jupyter notebook data/Simulated_DS.ipynb

Run all cells to generate smart_study_planner_dataset.csv

Step 3: Train Models

cd src
python main.py

This will train all models and save them to models/ directory.
Time: 10-30 minutes

Step 4: Run Web Application

cd ..
streamlit run app.py

Application opens at: http://localhost:8501

═══════════════════════════════════════════════════════════════════════════════

6. PROJECT INFORMATION

Dataset:
• File: data/smart_study_planner_dataset.csv
• Samples: 1000
• Features: 6 (course_difficulty, credit_hours, number_of_courses, previous_grade, days_until_exam, daily_available_hours)
• Target: required_study_hours (weekly)

Models:
• Baseline: Linear Regression, Decision Tree
• Ensemble (6): Random Forest, Gradient Boosting, AdaBoost, Bagging, Voting, Stacking
• Final Model: Voting Regressor (R²=0.933)

Key Files:
• Dataset: data/smart_study_planner_dataset.csv
• Trained Model: models/final_voting_regressor.pkl
• Feature Scaler: models/feature_scaler.pkl
• Web App: app.py
• Main Pipeline: src/main.py

═══════════════════════════════════════════════════════════════════════════════
