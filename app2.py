# === PERFORMANCE OPTIMIZATIONS ===

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

from unsupervised_learning import UnsupervisedLearningAnalyzer, integrate_unsupervised_learning
from model_utils import load_model, LabelEncoderWrapper
from db import connect_to_db, create_users_table, add_user, authenticate_user, initialize_database
from feedback_utils import generate_feedback
from admin import admin_dashboard, manage_users, all_predictions



# Rest of your app code...
# Import augmentation functions
try:
    from augment_data import (
        augment_dataset_comprehensive, 
        retrain_with_comprehensive_data,
        enhanced_classify_grade,
        validate_comprehensive_grade_distribution
    )
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    st.warning("Data augmentation module not found. Some features may be limited.")

try:
    from unsupervised_learning import UnsupervisedLearningAnalyzer, integrate_unsupervised_learning
    UNSUPERVISED_AVAILABLE = True
except ImportError:
    UNSUPERVISED_AVAILABLE = False
    st.warning("Unsupervised learning module not found. Some advanced features may be limited.")



# 🚀 OPTIMIZATION 1: Set page config FIRST and only once
st.set_page_config(page_title="EduPredict", layout="wide")

# 🚀 OPTIMIZATION 2: Cache everything that's expensive
@st.cache_resource
def initialize_app():
    """Initialize database and model - runs only once per session"""
    initialize_database()
    create_users_table()
    
    # Try to load the comprehensive model first, fallback to original
    model_files = [
        "comprehensive_balanced_grade_model.pkl",
        "balanced_grade_model.pkl",
        "grade_model.pkl"  # Another fallback
    ]
    
    model = None
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                model = load_model(model_file)
                st.session_state.current_model_file = model_file
                break
        except Exception as e:
            continue
    
    if model is None:
        st.error("No valid model file found. Please train a model first.")
        st.stop()
    
    return model

@st.cache_resource
def load_unsupervised_analyzer():
    """Load and cache unsupervised learning analyzer"""
    if UNSUPERVISED_AVAILABLE:
        analyzer = UnsupervisedLearningAnalyzer()
        if analyzer.load_and_prepare_data():
            analyzer.perform_student_clustering()
            analyzer.detect_anomalies()
            analyzer.perform_pca_analysis()
            return analyzer
    return None


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_user_results(user_email):
    """Cache user CSV data for 1 minute"""
    csv_path = f"results_{user_email}.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def get_custom_css():
    """Cache CSS string - never changes"""
    return """
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #f0f4f8;
            color: #333333;
        }
        .main {
            background-color: #f0f4f8;
        }
        .stButton>button {
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 0.6em 2em;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            padding: 12px;
            color: #4a5568;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #667eea;
            color: white;
            border-radius: 8px 8px 0 0;
        }
        .admin-section {
            background-color: #fff5f5;
            border-left: 4px solid #f56565;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }
    </style>
    """

# 🚀 OPTIMIZATION 3: Helper function for feature engineering
def add_engineered_features(df):
    """Add consistent feature engineering across the app"""
    df = df.copy()
    df['Study_Attendance_Interaction'] = df['Hours_Studied'] * df['Attendance']
    df['Previous_Scores_Binned'] = pd.cut(
        df['Previous_Scores'], 
        bins=[0, 60, 80, 100], 
        labels=['Low', 'Medium', 'High']
    ).astype(str)
    return df

# 🚀 OPTIMIZATION 4: Initialize once at startup
def ensure_model_loaded():
    """Ensure model is loaded with fallback"""
    if 'model' not in st.session_state:
        with st.spinner("Loading EduPredict..."):
            st.session_state.model = initialize_app()
    return st.session_state.model

# Helper function for data augmentation
def check_data_quality(csv_path="data.csv"):
    """Check if original data needs augmentation"""
    if not os.path.exists(csv_path):
        return False, "Original data file not found"
    
    try:
        df = pd.read_csv(csv_path)
        if 'Exam_Score' not in df.columns:
            return False, "Exam_Score column missing"
        
        # Check grade distribution
        df['Grade_Class'] = df['Exam_Score'].apply(enhanced_classify_grade)
        grade_counts = df['Grade_Class'].value_counts()
        
        # Check for missing grades
        expected_grades = ['First Class', 'Second Class Upper', 'Second Class Lower', 
                          'Third Class', 'Pass', 'Fail']
        missing_grades = [grade for grade in expected_grades if grade not in grade_counts.index]
        
        if missing_grades:
            return False, f"Missing grade categories: {', '.join(missing_grades)}"
        
        # Check for severe imbalance
        min_count = grade_counts.min()
        max_count = grade_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 20:
            return False, f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)"
        
        return True, "Data quality is acceptable"
        
    except Exception as e:
        return False, f"Error analyzing data: {str(e)}"

# Initialize app
if 'app_initialized' not in st.session_state:
    model = ensure_model_loaded()
    st.session_state.app_initialized = True
else:
    model = ensure_model_loaded()

# Apply CSS every time to ensure it's always loaded
st.markdown(get_custom_css(), unsafe_allow_html=True)

# 🚀 OPTIMIZATION 5: Streamlined auth function
def auth_app():
    st.title("EduPredict - Authentication")
    
    action = st.selectbox("Choose Action", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if action == "Sign Up":
        role = st.radio("Select Role", ["Student", "Educator"])
        if st.button("Sign Up"):
            if email and password:
                if add_user(email, password, role):
                    st.success("Account created successfully!")
                    st.session_state.update({
                        'logged_in': True,
                        'user_email': email,
                        'user_role': role
                    })
                    st.rerun()
                else:
                    st.error("Email already exists or registration failed.")
            else:
                st.error("Please fill in all fields.")
    else:
        if st.button("Login"):
            if email and password:
                user = authenticate_user(email, password)
                if user:
                    st.success(f"Welcome back!")
                    st.session_state.update({
                        'logged_in': True,
                        'user_email': user['email'],
                        'user_role': user['role']
                    })
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.error("Please enter both email and password.")

# 🚀 NEW: Data Management Section for Educators
def data_management_section():
    st.markdown('<div class="admin-section">', unsafe_allow_html=True)
    st.subheader("Data Management & Model Training")
    
    # Check current model info
    current_model = getattr(st.session_state, 'current_model_file', 'Unknown')
    st.info(f"Currently using model: **{current_model}**")
    
    # Data quality check
    st.markdown("### 📊 Data Quality Analysis")
    if st.button("Check Data Quality"):
        is_good, message = check_data_quality()
        if is_good:
            st.success(f"✅ {message}")
        else:
            st.warning(f"⚠️ {message}")
            st.markdown("**Recommendation:** Consider running data augmentation to improve model performance.")
    
    # Data augmentation section
    if AUGMENTATION_AVAILABLE:
        st.markdown("### Data Augmentation & Model Training")
        
        col1, col2 = st.columns(2)
        with col1:
            n_third_class = st.number_input("Third Class Records to Generate", 100, 1000, 500)
            n_pass = st.number_input("Pass Grade Records to Generate", 100, 1000, 400)
        with col2:
            n_fail = st.number_input("Fail Grade Records to Generate", 100, 1000, 300)
        
        if st.button("🔄 Run Data Augmentation & Retrain Model"):
            with st.spinner("Augmenting data and retraining model... This may take a few minutes."):
                try:
                    # Step 1: Augment data
                    st.info("Step 1: Creating augmented dataset...")
                    augmented_df = augment_dataset_comprehensive(
                        original_csv_path="data.csv",
                        output_csv_path="comprehensive_augmented_data.csv",
                        n_third_class=n_third_class,
                        n_pass=n_pass,
                        n_fail=n_fail
                    )
                    
                    # Step 2: Retrain model
                    st.info("Step 2: Retraining model with balanced data...")
                    retrain_with_comprehensive_data(
                        "comprehensive_augmented_data.csv",
                        "comprehensive_balanced_grade_model.pkl"
                    )
                    
                    # Step 3: Reload model in session
                    st.info("Step 3: Loading new model...")
                    st.session_state.model = load_model("comprehensive_balanced_grade_model.pkl")
                    st.session_state.current_model_file = "comprehensive_balanced_grade_model.pkl"
                    
                    st.success("🎉 Data augmentation and model retraining completed successfully!")
                    st.balloons()
                    
                    # Show summary
                    st.markdown("### 📋 Summary")
                    st.write(f"- **Original data**: data.csv")
                    st.write(f"- **Augmented data**: comprehensive_augmented_data.csv")
                    st.write(f"- **New model**: comprehensive_balanced_grade_model.pkl")
                    st.write(f"- **Total records**: {len(augmented_df):,}")
                    
                except Exception as e:
                    st.error(f"Error during augmentation/training: {str(e)}")
    
    else:
        st.warning("Data augmentation module not available. Please ensure augment_data.py is in the same directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add this new tab to your data management section for Educators
def unsupervised_analytics_section():
    """Advanced analytics section using unsupervised learning"""
    st.markdown('<div class="admin-section">', unsafe_allow_html=True)
    st.subheader("🧠 Advanced Student Analytics (Unsupervised Learning)")
    
    if not UNSUPERVISED_AVAILABLE:
        st.error("Unsupervised learning module not available. Please ensure unsupervised_learning.py is in the same directory.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Initialize analyzer
    with st.spinner("Loading advanced analytics..."):
        analyzer = load_unsupervised_analyzer()
    
    if analyzer is None:
        st.error("Could not initialize unsupervised learning analyzer.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create tabs for different analytics
    analytics_tabs = st.tabs(["👥 Student Clusters", "🚨 Anomaly Detection", "📊 PCA Analysis", "🎯 Insights"])
    
    # Tab 1: Student Clustering
    with analytics_tabs[0]:
        st.markdown("### 👥 Student Profile Clusters")
        st.info("Students are automatically grouped based on similar characteristics and performance patterns.")
        
        if analyzer.student_profiles:
            # Display cluster summary
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create cluster summary dataframe
                cluster_data = []
                for cluster_id, profile in analyzer.student_profiles.items():
                    cluster_data.append({
                        'Cluster': profile['name'],
                        'Description': profile['description'],
                        'Students': profile['size'],
                        'Avg Score': f"{profile['avg_exam_score']:.1f}%",
                        'Success Rate': analyzer.calculate_success_probability(cluster_id)
                    })
                
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df, use_container_width=True)
            
            with col2:
                # Cluster size pie chart
                cluster_sizes = [profile['size'] for profile in analyzer.student_profiles.values()]
                cluster_names = [profile['name'] for profile in analyzer.student_profiles.values()]
                
                fig_pie = px.pie(
                    values=cluster_sizes,
                    names=cluster_names,
                    title="Student Distribution by Cluster"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Detailed cluster analysis
            selected_cluster = st.selectbox(
                "Select cluster for detailed analysis:",
                options=list(analyzer.student_profiles.keys()),
                format_func=lambda x: analyzer.student_profiles[x]['name']
            )
            
            if selected_cluster is not None:
                profile = analyzer.student_profiles[selected_cluster]
                cluster_students = analyzer.df[analyzer.df['Student_Cluster'] == selected_cluster]
                
                st.markdown(f"#### 📋 {profile['name']} - Detailed Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Students", profile['size'])
                with col2:
                    st.metric("Avg Score", f"{profile['avg_exam_score']:.1f}%")
                with col3:
                    st.metric("Avg Study Hours", f"{profile['avg_hours_studied']:.1f}")
                with col4:
                    st.metric("Avg Attendance", f"{profile['avg_attendance']:.1f}%")
                
                # Grade distribution for this cluster
                grade_dist = cluster_students['Grade_Class'].value_counts()
                fig_bar = px.bar(
                    x=grade_dist.index,
                    y=grade_dist.values,
                    title=f"Grade Distribution - {profile['name']}",
                    labels={'x': 'Grade', 'y': 'Number of Students'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tab 2: Anomaly Detection
    with analytics_tabs[1]:
        st.markdown("### 🚨 Anomaly Detection")
        st.info("Identifies students with unusual patterns that may need special attention.")
        
        if hasattr(analyzer, 'anomalies') and analyzer.anomalies is not None:
            anomaly_count = sum(analyzer.df['Is_Anomaly'])
            total_students = len(analyzer.df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Students", total_students)
            with col2:
                st.metric("Anomalous Students", anomaly_count)
            with col3:
                st.metric("Anomaly Rate", f"{anomaly_count/total_students*100:.1f}%")
            
            if anomaly_count > 0:
                anomalous_students = analyzer.df[analyzer.df['Is_Anomaly'] == True]
                
                st.markdown("#### 🔍 Anomalous Student Characteristics")
                
                # Compare anomalous vs normal students
                normal_students = analyzer.df[analyzer.df['Is_Anomaly'] == False]
                
                comparison_data = {
                    'Metric': ['Avg Exam Score', 'Avg Study Hours', 'Avg Attendance', 'Avg Sleep Hours'],
                    'Anomalous Students': [
                        anomalous_students['Exam_Score'].mean(),
                        anomalous_students['Hours_Studied'].mean(),
                        anomalous_students['Attendance'].mean(),
                        anomalous_students['Sleep_Hours'].mean()
                    ],
                    'Normal Students': [
                        normal_students['Exam_Score'].mean(),
                        normal_students['Hours_Studied'].mean(),
                        normal_students['Attendance'].mean(),
                        normal_students['Sleep_Hours'].mean()
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualization of anomalies
                if 'PCA_Component_1' in analyzer.df.columns:
                    fig_scatter = px.scatter(
                        analyzer.df,
                        x='PCA_Component_1',
                        y='PCA_Component_2',
                        color='Is_Anomaly',
                        title="Student Distribution with Anomalies Highlighted",
                        labels={'PCA_Component_1': 'First Principal Component', 
                               'PCA_Component_2': 'Second Principal Component'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Anomaly detection not available. Please run the analysis first.")
    
    # Tab 3: PCA Analysis
    with analytics_tabs[2]:
        st.markdown("### 📊 Principal Component Analysis (PCA)")
        st.info("Dimensionality reduction to visualize student data patterns in 2D space.")
        
        if 'PCA_Component_1' in analyzer.df.columns:
            # PCA scatter plot colored by grade
            fig_pca = px.scatter(
                analyzer.df,
                x='PCA_Component_1',
                y='PCA_Component_2',
                color='Grade_Class',
                title="Student Distribution in PCA Space (Colored by Grade)",
                labels={
                    'PCA_Component_1': 'First Principal Component',
                    'PCA_Component_2': 'Second Principal Component'
                }
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # PCA scatter plot colored by cluster
            fig_cluster = px.scatter(
                analyzer.df,
                x='PCA_Component_1',
                y='PCA_Component_2',
                color='Student_Cluster',
                title="Student Distribution in PCA Space (Colored by Cluster)",
                labels={
                    'PCA_Component_1': 'First Principal Component',
                    'PCA_Component_2': 'Second Principal Component'
                }
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("PCA analysis not available. Please run the analysis first.")
    
    # Tab 4: Insights and Recommendations
    with analytics_tabs[3]:
        st.markdown("### 🎯 Key Insights & Recommendations")
        
        if analyzer.student_profiles:
            st.markdown("#### 📈 Performance Insights")
            
            # Find best and worst performing clusters
            cluster_performance = [(cid, profile['avg_exam_score']) for cid, profile in analyzer.student_profiles.items()]
            best_cluster = max(cluster_performance, key=lambda x: x[1])
            worst_cluster = min(cluster_performance, key=lambda x: x[1])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"**Best Performing Group:** {analyzer.student_profiles[best_cluster[0]]['name']}")
                st.write(f"Average Score: {best_cluster[1]:.1f}%")
                st.write("Key characteristics:")
                best_students = analyzer.df[analyzer.df['Student_Cluster'] == best_cluster[0]]
                st.write(f"• Study Hours: {best_students['Hours_Studied'].mean():.1f} per week")
                st.write(f"• Attendance: {best_students['Attendance'].mean():.1f}%")
                st.write(f"• Sleep: {best_students['Sleep_Hours'].mean():.1f} hours")
            
            with col2:
                st.error(f"**Needs Attention:** {analyzer.student_profiles[worst_cluster[0]]['name']}")
                st.write(f"Average Score: {worst_cluster[1]:.1f}%")
                st.write("Areas for improvement:")
                worst_students = analyzer.df[analyzer.df['Student_Cluster'] == worst_cluster[0]]
                st.write(f"• Study Hours: {worst_students['Hours_Studied'].mean():.1f} per week")
                st.write(f"• Attendance: {worst_students['Attendance'].mean():.1f}%")
                st.write(f"• Sleep: {worst_students['Sleep_Hours'].mean():.1f} hours")
            
            # Overall recommendations
            st.markdown("#### 💡 Action Items")
            
            total_at_risk = sum(1 for profile in analyzer.student_profiles.values() 
                              if profile['avg_exam_score'] < 50)
            
            if total_at_risk > 0:
                st.warning(f"🚨 {total_at_risk} student group(s) are performing below average and need intervention.")
            
            if hasattr(analyzer, 'anomalies') and analyzer.anomalies is not None:
                anomaly_count = sum(analyzer.df['Is_Anomaly'])
                if anomaly_count > 0:
                    st.info(f"📊 {anomaly_count} students show unusual patterns and may benefit from personalized attention.")
            
            st.success("✨ Use these insights to provide targeted support and improve overall class performance!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 🚀 OPTIMIZATION 6
def main_app():
    st.sidebar.title("EduPredict")
    st.sidebar.write(f"👤 {st.session_state.user_email} ({st.session_state.user_role})")

    if st.sidebar.button("Log Out"):
        # Clear session state efficiently
        for key in ['logged_in', 'user_email', 'user_role', 'last_prediction']:
            st.session_state.pop(key, None)
        st.rerun()

    # Admin UI for Educators
    if st.session_state.user_role == "Educator":
        admin_section = st.sidebar.selectbox("Admin Options", 
            ["Dashboard", "Manage Users", "All Predictions", "Advanced Analytics"])
        
        if admin_section == "Dashboard":
            admin_dashboard()
            return
        elif admin_section == "Manage Users":
            manage_users()
            return
        elif admin_section == "All Predictions":
            all_predictions()
            return
        elif admin_section == "Advanced Analytics":
            unsupervised_analytics_section()
            return

    st.title("🎓 Grade Prediction System")
    tabs = st.tabs(["Predict Grade", "Results", "What-If", "Feedback"])

    # --- Tab: Predict Grade ---
    with tabs[0]:
        st.subheader("🧾 Student Information")

        # Academic Info
        st.markdown("### Academic Information")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("Weekly Study Hours", 0, 50, 10)
            previous_scores = st.number_input("Previous scores (%)", 0.0, 100.0, 70.0)
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], key="teacher_quality")
        with col2:
            attendance = st.slider("Class Attendance (%)", 0, 100, 85)
            tutoring_sessions = st.number_input("Tutoring Sessions per Week", 0, 10, 2)
            school_type = st.selectbox("School Type", ["Private", "Public"], key="school_type")

        # Personal Info
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            sleep_hours = st.number_input("Average Sleep Hours per Day", 0, 12, 7)
            internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"])
        with col2:
            learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
            physical_activity = st.selectbox("Physical Activity Level", [0, 1, 2, 3, 4, 5])
            distance_from_home = st.selectbox("Distance from Home to School (km)", ["Near", "Moderate", "Far"])

        # Environmental Factors
        st.markdown("### Environmental Factors")
        col1, col2 = st.columns(2)
        with col1:
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
            parental_education_level = st.selectbox("Parental Education Level", ["College", "High School", "Postgraduate"])
        with col2:
            family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
            access_to_resources = st.selectbox("Access to Learning Resources", ["Low", "Medium", "High"])

        # Behavioral Factors
        st.markdown("### Behavioral Factors")
        col1, col2 = st.columns(2)
        with col1:
            motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
            peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        with col2:
            extracurricular_activities = st.selectbox("Extracurricular Activities", ["No", "Yes"])

        # 🔧 FIXED: Prediction generation
        if st.button("Generate Prediction", key="Generate_Prediction"):
            # Create input_dict with current form values
            input_dict = {
                'Hours_Studied': study_hours,
                'Previous_Scores': previous_scores,
                'Attendance': attendance,
                'Tutoring_Sessions': tutoring_sessions,
                'School_Type': school_type,
                'Teacher_Quality': teacher_quality,
                'Gender': gender,
                'Sleep_Hours': sleep_hours,
                'Internet_Access': internet_access,
                'Learning_Disabilities': learning_disabilities,
                'Physical_Activity': physical_activity,
                'Distance_from_Home': distance_from_home,
                'Parental_Involvement': parental_involvement,
                'Parental_Education_Level': parental_education_level,
                'Family_Income': family_income,
                'Access_to_Resources': access_to_resources,
                'Motivation_Level': motivation_level,
                'Peer_Influence': peer_influence,
                'Extracurricular_Activities': extracurricular_activities,
            }

            # Validate that all required fields are filled
            if all(v is not None and v != "" for v in input_dict.values()):
                input_df = pd.DataFrame([input_dict])
                
                # Apply feature engineering
                input_df = add_engineered_features(input_df)

                try:
                    # Predict
                    prediction_proba = model.predict_proba(input_df)[0]
                    predicted_idx = np.argmax(prediction_proba)
                    predicted_class = model.classes_[predicted_idx]
                    confidence = round(prediction_proba[predicted_idx] * 100, 2)

                    st.markdown(f"### Predicted Grade Classification: **{predicted_class}**")
                    st.markdown(f"**Confidence Score:** {confidence}%")

                    # Show probability distribution
                    dist = dict(zip(model.classes_, (prediction_proba * 100).round(2)))
                    fig = px.bar(
                        x=list(dist.keys()),
                        y=list(dist.values()),
                        labels={"x": "Grade Classification", "y": "Probability (%)"},
                        title="Prediction Probability Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show grade interpretation
                    grade_info = {
                        'First Class': '70-100% (Excellent Performance)',
                        'Second Class Upper': '60-69% (Very Good Performance)',
                        'Second Class Lower': '50-59% (Good Performance)',
                        'Third Class': '45-49% (Satisfactory Performance)',
                        'Pass': '40-44% (Marginal Pass)',
                        'Fail': '0-39% (Below Pass Standard)'
                    }
                    
                    if predicted_class in grade_info:
                        st.info(f"**Grade Meaning:** {grade_info[predicted_class]}")

                    # Show feedback tips
                    st.markdown("### Suggested Improvements:")
                    feedback_tips = generate_feedback(predicted_class, input_dict)
                    if isinstance(feedback_tips, list):
                        for tip in feedback_tips:
                            st.write(f"- {tip}")
                    else:
                        st.write(feedback_tips)

                    # Save prediction result
                    result_row = input_dict.copy()
                    result_row.update({
                        "Predicted_Grade": predicted_class,
                        "Confidence": confidence,
                        "Previous_Scores_Binned": input_df["Previous_Scores_Binned"].iloc[0],
                        "Study_Attendance_Interaction": input_df["Study_Attendance_Interaction"].iloc[0],
                        "Timestamp": datetime.datetime.now().isoformat()
                    })

                    # Append result to CSV
                    csv_path = f"results_{st.session_state.user_email}.csv"
                    if os.path.exists(csv_path):
                        df_results = pd.read_csv(csv_path)
                        df_results = pd.concat([df_results, pd.DataFrame([result_row])], ignore_index=True)
                    else:
                        df_results = pd.DataFrame([result_row])

                    df_results.to_csv(csv_path, index=False)
                    st.success("Prediction saved successfully!")

                    # Store in session state for feedback tab
                    st.session_state.last_prediction = {
                        'input_dict': input_dict,
                        'predicted_class': predicted_class,
                        'confidence': confidence
                    }

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error("Please check your model file and ensure all features are properly configured.")
            
            else:
                st.error("Please fill in all fields before generating a prediction.")

    # 🚀 OPTIMIZATION 7: Optimized Results Tab
    with tabs[1]:
        st.subheader(" Prediction History")
        
        # Use cached data loading
        df = load_user_results(st.session_state.user_email)
        
        if df is not None and len(df) > 0:
            # Show only recent results by default for speed
            recent_df = df.tail(20)
            st.dataframe(recent_df, use_container_width=True)
            
            # Show grade distribution if we have the column
            if 'Predicted_Grade' in df.columns:
                st.markdown("### Your Grade Distribution")
                grade_counts = df['Predicted_Grade'].value_counts()
                fig = px.pie(values=grade_counts.values, names=grade_counts.index, 
                           title="Distribution of Your Predicted Grades")
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(df))
            with col2:
                if len(df) > 0 and 'Confidence' in df.columns:
                    avg_confidence = df['Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            with col3:
                if 'Predicted_Grade' in df.columns:
                    most_common = df['Predicted_Grade'].mode().iloc[0] if len(df) > 0 else "N/A"
                    st.metric("Most Common Grade", most_common)
            
            st.download_button(
                label="Download All Results",
                data=df.to_csv(index=False),
                file_name=f"predictions_{st.session_state.user_email}.csv",
                mime='text/csv'
            )
        else:
            st.info("No predictions yet. Make your first prediction!")

    # 🚀 OPTIMIZATION 8: Optimized What-If Analysis
    with tabs[2]:
        st.subheader("🔄 What-If Analysis")
        
        df_results = load_user_results(st.session_state.user_email)
        
        if df_results is not None and len(df_results) > 0:
            # Validate required columns
            required_cols = ['Predicted_Grade', 'Confidence', 'Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions', 'Sleep_Hours']
            if not all(col in df_results.columns for col in required_cols):
                st.error("Data format issue. Please regenerate predictions to use What-If analysis.")
            else:
                # Use selectbox with better formatting
                record_options = [f"Record {i+1} - {row['Predicted_Grade']} ({row['Confidence']:.1f}%)" 
                                for i, (_, row) in enumerate(df_results.iterrows())]
                
                selected_display = st.selectbox("Choose a record to modify:", record_options)
                record_idx = int(selected_display.split()[1]) - 1
                
                selected_record = df_results.iloc[record_idx].copy()
                
                # Show compact record info
                st.info(f"**Original Prediction:** {selected_record['Predicted_Grade']} "
                       f"({selected_record['Confidence']:.1f}% confidence)")

                # Modification controls in columns for better layout
                st.markdown("#### 🔧 Modify Key Factors")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_study_hours = st.slider("Study Hours", 0, 50, int(selected_record["Hours_Studied"]))
                    new_attendance = st.slider("Attendance (%)", 0, 100, int(selected_record["Attendance"]))
                with col2:
                    new_previous_scores = st.number_input("Previous Scores (%)", 0.0, 100.0, float(selected_record["Previous_Scores"]))
                    new_tutoring_sessions = st.number_input("Tutoring Sessions", 0, 10, int(selected_record["Tutoring_Sessions"]))
                with col3:
                    new_sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0, float(selected_record["Sleep_Hours"]))

                # Update record with new values
                selected_record["Hours_Studied"] = new_study_hours
                selected_record["Previous_Scores"] = new_previous_scores
                selected_record["Attendance"] = new_attendance
                selected_record["Tutoring_Sessions"] = new_tutoring_sessions
                selected_record["Sleep_Hours"] = new_sleep_hours
                
                # Apply feature engineering using helper function
                temp_df = pd.DataFrame([selected_record])
                temp_df = add_engineered_features(temp_df)
                selected_record = temp_df.iloc[0]

                # Predict with updated values
                try:
                    features = model.named_steps['preprocessor'].feature_names_in_
                    input_df = pd.DataFrame([selected_record[features]])
                    
                    prediction_proba = model.predict_proba(input_df)[0]
                    predicted_idx = np.argmax(prediction_proba)
                    predicted_class = model.classes_[predicted_idx]
                    confidence = round(prediction_proba[predicted_idx] * 100, 2)

                    # Show results
                    col1, col2 = st.columns(2)
                    with col1:
                        original_confidence = df_results.iloc[record_idx]['Confidence']
                        confidence_change = confidence - original_confidence
                        st.metric(
                            "New Prediction", 
                            predicted_class,
                            delta=f"{confidence_change:+.1f}% confidence"
                        )
                    with col2:
                        st.metric("Confidence", f"{confidence}%")

                    # Compact probability chart
                    if st.checkbox("Show probability breakdown"):
                        prob_data = pd.DataFrame({
                            'Grade': model.classes_,
                            'Probability': prediction_proba * 100
                        })
                        fig = px.bar(prob_data, x='Grade', y='Probability', 
                                   title="Updated Prediction Probabilities")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("No prediction history found. Make a prediction first!")

    # 🚀 OPTIMIZATION 9: Fixed Feedback Tab (proper indentation)
    with tabs[3]:
        st.markdown("## 📝 Personalized Feedback")
        
        st.info("In this section, you'll receive tailored advice based on your prediction and input details.")
        
        # Check if we have a recent prediction in session state
        if 'last_prediction' in st.session_state:
            prediction_data = st.session_state.last_prediction
            input_dict = prediction_data['input_dict']
            predicted_class = prediction_data['predicted_class']
            confidence = prediction_data['confidence']
            
            st.markdown(f"### Last Prediction: **{predicted_class}** ({confidence}% confidence)")
            
            # Generate personalized feedback
            actionable_feedback = generate_feedback(predicted_class, input_dict)
            
            st.markdown("### 🎯 Here's what we suggest:")
            
            # Handle different feedback formats
            if isinstance(actionable_feedback, str):
                # Split by paragraphs and display with separators
                paragraphs = actionable_feedback.strip().split("\n\n")
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Only show non-empty paragraphs
                        st.write(paragraph.strip())
                        if i < len(paragraphs) - 1:  # Don't add separator after last paragraph
                            st.markdown("---")
                            
            elif isinstance(actionable_feedback, list):
                for tip in actionable_feedback:
                    st.write(f"- {tip}")
            else:
                st.warning("⚠️ Feedback format is unrecognized. Please check the generate_feedback function.")
                
            # Option to clear feedback
            if st.button("Clear Feedback History"):
                if 'last_prediction' in st.session_state:
                    del st.session_state.last_prediction
                st.rerun()
                
        else:
            st.warning("⚠️ Please generate a prediction first to view personalized feedback.")
            st.info("Go to the 'Predict Grade' tab and click 'Generate Prediction' to get started!")

# 🚀 OPTIMIZATION 10: Main execution
def main():
    # Initialize session state once
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        auth_app()
    else:
        main_app()

if __name__ == "__main__":
    main()