# =====================================================
# SMART STUDY PLANNER - ENTERPRISE WEB APPLICATION
# =====================================================
"""
Professional Streamlit web application for Smart Study Planner.
Features:
- Real-time ML predictions
- Interactive visualizations
- Calendar view
- Multiple export formats
- Model confidence display
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Page configuration
st.set_page_config(
    page_title="Smart Study Planner",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL LOADING
# =====================================================
@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        model_paths = [
            os.path.join("models", "final_voting_regressor.pkl"),
            os.path.join("models", "final_tuned_model.pkl")
        ]
        scaler_path = os.path.join("models", "feature_scaler.pkl")
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                break
        
        if model is None:
            raise FileNotFoundError("No model file found")
        
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ **Model files not found!**\n\nPlease run the training pipeline:\n```bash\ncd src\npython main.py\n```")
        st.stop()

FEATURE_ORDER = [
    "course_difficulty", "credit_hours", "number_of_courses",
    "previous_grade", "days_until_exam", "daily_available_hours"
]

def predict_study_hours(model, scaler, courses_data, apply_constraint=False, max_weekly_hours=None):
    """Predict study hours with proper feature handling and optional available time constraint"""
    X = courses_data[FEATURE_ORDER]
    X_array = X.values if isinstance(X, pd.DataFrame) else X.to_numpy()
    X_scaled = scaler.transform(X_array)
    predictions = model.predict(X_scaled)
    predictions = np.clip(predictions, 1, None)
    
    # Apply available time constraint if requested
    if apply_constraint and max_weekly_hours is not None:
        total_predicted = np.sum(predictions)
        
        if total_predicted > max_weekly_hours:
            # Scale down predictions proportionally to fit available time
            scale_factor = max_weekly_hours / total_predicted
            predictions = predictions * scale_factor
            predictions = np.clip(predictions, 1, None)  # Ensure minimum 1 hour per course
    
    return predictions

def generate_schedule(courses_data, predictions):
    """Generate weekly and daily study schedule with intelligent distribution"""
    courses_data = courses_data.copy()
    courses_data['required_study_hours'] = predictions
    
    weekly_plan = {row['course_name']: round(row['required_study_hours'], 2) 
                   for _, row in courses_data.iterrows()}
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_schedule = {day: {} for day in days}
    
    # Distribute study hours based on urgency and difficulty
    # Lower days_until_exam = more urgent = more study time needed now
    for _, row in courses_data.iterrows():
        course_name = row['course_name']
        weekly_hours = row['required_study_hours']
        days_until_exam = row['days_until_exam']
        difficulty = row['course_difficulty']
        
        # Distribute hours across the week based on urgency
        # More urgent courses (fewer days until exam) get more hours in early days
        if days_until_exam <= 7:
            # Very urgent: concentrate hours in first 5 days (weekdays)
            distribution = [0.20, 0.20, 0.20, 0.20, 0.20, 0.00, 0.00]
        elif days_until_exam <= 14:
            # Urgent: focus on weekdays, some weekend
            distribution = [0.18, 0.18, 0.18, 0.18, 0.18, 0.05, 0.05]
        elif days_until_exam <= 30:
            # Moderate: balanced weekdays, moderate weekend
            distribution = [0.15, 0.15, 0.15, 0.15, 0.15, 0.12, 0.13]
        else:
            # Not urgent: evenly distributed
            distribution = [1/7] * 7
        
        # Apply difficulty adjustment for hard courses (more consistent distribution)
        if difficulty >= 4:
            # Hard courses: blend with even distribution for consistency
            even_dist = [1/7] * 7
            distribution = [h * 0.9 + even_dist[i] * 0.1 for i, h in enumerate(distribution)]
        
        # Normalize to ensure total = 1.0 (safety check)
        total = sum(distribution)
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            distribution = [h / total for h in distribution]
        
        # Assign hours to each day
        for i, day in enumerate(days):
            daily_schedule[day][course_name] = round(weekly_hours * distribution[i], 2)
    
    return weekly_plan, daily_schedule, courses_data

# =====================================================
# MAIN APPLICATION
# =====================================================
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📚 Smart Study Planner</h1>
            <p style="font-size: 1.2rem;">AI-Powered Personalized Study Schedule Generator</p>
        </div>
    """, unsafe_allow_html=True)
    
    model, scaler = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        daily_hours = st.slider("Daily Available Study Hours", 1.0, 6.0, 3.5, 0.5)
        st.caption("⚠️ Valid range: 1-6 hours per day")
        st.markdown(f"**📊 Weekly Capacity:** {daily_hours * 7:.1f} hours/week")
        st.markdown("---")
        st.info("""
        **How Available Hours Work:**
        
        This sets your **maximum** weekly study capacity. If predictions exceed this limit, they will be automatically scaled down proportionally.
        
        **Example:** If you have 6 hours/day = 42 hours/week, but the model predicts 60 hours/week total, predictions will be scaled to fit within 42 hours.
        """)
        st.markdown("---")
        st.info("**Voting Regressor**\n\nCombines multiple ensemble models for accurate predictions.")
    
    # Main content
    st.header("📝 Add Your Courses")
    
    # Info box with valid ranges
    with st.expander("ℹ️ Valid Input Ranges (Click to expand)", expanded=False):
        st.markdown("""
        **Please ensure all inputs are within these valid ranges for accurate predictions:**
        
        - **Course Difficulty:** 1-5 (1=Easy, 5=Very Difficult)
        - **Credit Hours:** 2-6 hours
        - **Previous Grade:** 50-100% (minimum passing grade)
        - **Days Until Exam:** 1-120 days
        - **Daily Available Hours:** 1-6 hours per day
        - **Number of Courses:** 3-8 courses
        
        ⚠️ **Note:** Inputs outside these ranges may produce unreliable predictions as the model was trained on data within these constraints.
        """)
    
    num_courses = st.number_input("Number of Courses", 3, 8, 3, 1)
    st.caption("⚠️ Valid range: 3-8 courses")
    
    # Validation function
    def validate_inputs(courses_data):
        """Validate all inputs are within expected ranges and check for edge cases"""
        errors = []
        warnings = []
        
        # Check for duplicate course names
        course_names = [course['course_name'] for course in courses_data]
        duplicates = [name for name in set(course_names) if course_names.count(name) > 1]
        if duplicates:
            errors.append(f"Duplicate course names found: {', '.join(duplicates)}. Please use unique names for each course.")
        
        # Check for empty course names
        empty_names = [i+1 for i, course in enumerate(courses_data) if not course['course_name'] or course['course_name'].strip() == '']
        if empty_names:
            errors.append(f"Course(s) {', '.join(map(str, empty_names))} have empty names. Please enter a course name.")
        
        for idx, course in enumerate(courses_data):
            course_num = idx + 1
            course_name = course['course_name']
            
            # Basic range validation
            if not (50 <= course['previous_grade'] <= 100):
                errors.append(f"{course_name} (Course {course_num}): Previous Grade must be between 50-100% (current: {course['previous_grade']}%)")
            if not (1 <= course['daily_available_hours'] <= 6):
                errors.append(f"{course_name} (Course {course_num}): Daily Available Hours must be between 1-6 (current: {course['daily_available_hours']})")
            if not (2 <= course['credit_hours'] <= 6):
                errors.append(f"{course_name} (Course {course_num}): Credit Hours must be between 2-6 (current: {course['credit_hours']})")
            if not (1 <= course['course_difficulty'] <= 5):
                errors.append(f"{course_name} (Course {course_num}): Difficulty must be between 1-5 (current: {course['course_difficulty']})")
            if not (1 <= course['days_until_exam'] <= 120):
                errors.append(f"{course_name} (Course {course_num}): Days Until Exam must be between 1-120 (current: {course['days_until_exam']})")
            
            # Edge case warnings
            if course['days_until_exam'] <= 2:
                warnings.append(f"⚠️ {course_name} has exam in {course['days_until_exam']} day(s) - very urgent!")
            if course['days_until_exam'] >= 100:
                warnings.append(f"ℹ️ {course_name} has exam in {course['days_until_exam']} days - not urgent, predictions may be lower")
            if course['previous_grade'] <= 55:
                warnings.append(f"⚠️ {course_name} has low previous grade ({course['previous_grade']}%) - may need more study time")
        
        return errors, warnings
    
    courses = []
    for i in range(num_courses):
        with st.expander(f"📖 Course {i+1}", expanded=(i < 2)):
            col1, col2 = st.columns(2)
            with col1:
                course_name = st.text_input(f"Course Name", f"Course {i+1}", key=f"name_{i}")
                course_difficulty = st.slider("Difficulty (1-5)", 1, 5, 3, key=f"diff_{i}", 
                                             help="1=Easy, 5=Very Difficult")
                credit_hours = st.slider("Credit Hours", 2, 6, 4, key=f"cred_{i}", 
                                        help="Valid range: 2-6 credit hours")
            with col2:
                previous_grade = st.slider("Previous Grade (%)", 50, 100, 75, key=f"grade_{i}", 
                                          help="Valid range: 50-100% (minimum passing grade)")
                days_until_exam = st.slider("Days Until Exam", 1, 120, 30, key=f"days_{i}", 
                                           help="Valid range: 1-120 days")
            
            courses.append({
                'course_name': course_name,
                'course_difficulty': course_difficulty,
                'credit_hours': credit_hours,
                'number_of_courses': num_courses,
                'previous_grade': previous_grade,
                'days_until_exam': days_until_exam,
                'daily_available_hours': daily_hours
            })
    
    st.markdown("---")
    
    if st.button("🚀 Generate Study Schedule", type="primary", use_container_width=True):
        courses_df = pd.DataFrame(courses)
        
        # Validate inputs before prediction
        validation_errors, validation_warnings = validate_inputs(courses)
        if validation_errors:
            st.error("❌ **Input Validation Errors:**\n\n" + "\n".join(f"• {err}" for err in validation_errors))
            st.warning("⚠️ **Please correct the above errors before generating the schedule.**\n\nInvalid inputs may produce unreliable predictions as the model was trained on data within specific ranges.")
            st.info("💡 **Tip:** Use the info box above to see all valid input ranges.")
            st.stop()
        
        # Show warnings if any
        if validation_warnings:
            st.warning("⚠️ **Input Warnings:**\n\n" + "\n".join(f"• {warn}" for warn in validation_warnings))
        
        # All inputs valid - proceed with prediction
        st.success("✅ All inputs are valid!")
        
        # Calculate available time
        max_weekly_hours = daily_hours * 7
        
        with st.spinner("🤖 Predicting study hours using ML model..."):
            # Predict study hours
            predictions = predict_study_hours(model, scaler, courses_df, apply_constraint=False)
            original_total = np.sum(predictions)
            
            # Check for unreasonable predictions
            min_per_course = 1.0  # Minimum 1 hour per course
            max_reasonable_per_course = 50.0  # Maximum reasonable hours per course per week
            
            if np.any(predictions > max_reasonable_per_course):
                st.warning(f"⚠️ **Unusually High Predictions Detected**\n\nSome courses have predictions > {max_reasonable_per_course} hours/week. This may indicate unrealistic input combinations.")
            
            if np.any(predictions < min_per_course):
                st.warning(f"⚠️ **Very Low Predictions Detected**\n\nSome courses have predictions < {min_per_course} hour/week. This may indicate the course is too easy or exam is very far away.")
            
            # Apply constraint if predictions exceed available time
            if original_total > max_weekly_hours:
                scale_factor = max_weekly_hours / original_total
                
                # Check if scaling is too extreme (less than 10% of original)
                if scale_factor < 0.1:
                    st.error(f"""
                    ❌ **Impossible Schedule**
                    
                    The model predicted **{original_total:.1f} hours/week**, but you only have **{max_weekly_hours:.1f} hours/week** available.
                    
                    This would require scaling down by **{scale_factor:.1%}**, which is unrealistic.
                    
                    **This schedule is not feasible with your current constraints.**
                    
                    **Solutions:**
                    - Reduce number of courses (currently {len(courses)})
                    - Increase daily available hours (currently {daily_hours} hours/day)
                    - Prioritize only the most urgent courses
                    """)
                    st.stop()
                
                predictions = predictions * scale_factor
                predictions = np.clip(predictions, 1, None)  # Ensure minimum 1 hour per course
                
                # Show warning about scaling
                st.warning(f"""
                ⚠️ **Time Constraint Applied**
                
                The model predicted **{original_total:.1f} hours/week** total, but you only have **{max_weekly_hours:.1f} hours/week** available ({daily_hours} hours/day × 7 days).
                
                Predictions have been scaled down by **{scale_factor:.1%}** to fit your available time.
                
                **Recommendation:** Consider reducing the number of courses, increasing your daily available hours, or prioritizing courses with exams coming soon.
                """)
            elif original_total > max_weekly_hours * 0.9:
                # Close to limit (within 90%)
                st.info(f"ℹ️ **Near Capacity**\n\nYour schedule uses {original_total:.1f} of {max_weekly_hours:.1f} available hours ({original_total/max_weekly_hours:.1%}). Consider leaving some buffer time.")
        
        weekly_plan, daily_schedule, courses_df = generate_schedule(courses_df, predictions)
        
        # Check if daily schedule exceeds daily available hours
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_totals_check = {}
        days_exceeding = []
        for day in days:
            daily_total = sum(daily_schedule[day].values())
            daily_totals_check[day] = daily_total
            if daily_total > daily_hours * 1.05:  # Allow 5% tolerance for rounding
                days_exceeding.append((day, daily_total))
        
        if days_exceeding:
            exceeding_msg = "\n".join([f"  • {day}: {hours:.1f} hours (available: {daily_hours} hours)" for day, hours in days_exceeding])
            st.warning(f"""
            ⚠️ **Some Days Exceed Available Time**
            
            {exceeding_msg}
            
            This may occur due to urgency-based distribution concentrating hours on weekdays. Consider adjusting your schedule or available hours.
            """)
        
        st.success("✅ Study schedule generated successfully!")
        st.markdown("---")
        
        # Metrics
        st.header("📊 Weekly Study Plan")
        total_hours = sum(weekly_plan.values())
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hours/Week", f"{total_hours:.1f}", 
                     delta=f"{max_weekly_hours - total_hours:.1f} available" if total_hours <= max_weekly_hours else None)
        with col2:
            st.metric("Avg Hours/Course", f"{total_hours/len(weekly_plan):.1f}")
        with col3:
            st.metric("Number of Courses", len(weekly_plan))
        with col4:
            st.metric("Hours/Day", f"{total_hours/7:.1f}", 
                     delta=f"Max: {daily_hours:.1f}/day" if total_hours/7 <= daily_hours else f"⚠️ Exceeds by {(total_hours/7) - daily_hours:.1f}")
        
        # Weekly chart
        fig_weekly = px.bar(
            x=list(weekly_plan.keys()),
            y=list(weekly_plan.values()),
            labels={'x': 'Course', 'y': 'Hours per Week'},
            title="📈 Weekly Study Hours Distribution",
            color=list(weekly_plan.values()),
            color_continuous_scale="Blues"
        )
        fig_weekly.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Course details
        st.subheader("📋 Course Details")
        display_df = courses_df[['course_name', 'course_difficulty', 'credit_hours', 
                                 'previous_grade', 'days_until_exam', 'required_study_hours']].copy()
        display_df.columns = ['Course', 'Difficulty', 'Credits', 'Previous Grade', 'Days to Exam', 'Study Hours/Week']
        display_df['Study Hours/Week'] = display_df['Study Hours/Week'].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Daily schedule
        st.header("📅 Daily Study Schedule")
        # Create DataFrame with days as rows, courses as columns
        daily_df = pd.DataFrame(daily_schedule).T.round(2)
        # Ensure all courses are included and in correct order
        course_order = [row['course_name'] for _, row in courses_df.iterrows()]
        # Reindex to ensure all courses appear as columns (in correct order)
        # If a course is missing, fill with 0
        daily_df = daily_df.reindex(columns=course_order, fill_value=0.0)
        # Verify all courses are present
        missing_courses = set(course_order) - set(daily_df.columns)
        if missing_courses:
            for course in missing_courses:
                daily_df[course] = 0.0
        # Reorder columns to match course_order
        daily_df = daily_df[course_order]
        daily_totals = daily_df.sum(axis=1)
        
        fig_daily = px.bar(
            x=daily_df.index,
            y=daily_totals,
            labels={'x': 'Day', 'y': 'Total Study Hours'},
            title="📊 Daily Study Hours Distribution",
            color=daily_totals,
            color_continuous_scale="Greens"
        )
        fig_daily.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Detailed schedule
        st.subheader("📝 Detailed Daily Schedule")
        daily_df.index.name = 'Day'
        st.dataframe(daily_df, use_container_width=True)
        
        # Download
        st.markdown("---")
        st.subheader("💾 Download Schedule")
        csv = courses_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"study_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
