# augment_data.py - Updated for seamless integration with app.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def enhanced_classify_grade(score):
    """
    Enhanced grade classification that matches your app's expectations
    """
    if score >= 70:
        return 'First Class'
    elif score >= 60:
        return 'Second Class Upper'
    elif score >= 50:
        return 'Second Class Lower'
    elif score >= 45:
        return 'Third Class'
    elif score >= 40:
        return 'Pass'
    else:
        return 'Fail'

def create_logically_consistent_dataset(original_df, n_third_class=500, n_pass=400, n_fail=300):
    """
    Create synthetic records with LOGICAL CONSISTENCY between features and outcomes
    """
    synthetic_records = []
    
    # Third Class students (45-49 exam scores)
    print(f"Creating {n_third_class} Third Class records (45-49%)...")
    for _ in range(n_third_class):
        base_score = np.random.randint(45, 50)
        previous_score = np.random.normal(base_score - 5, 8)
        previous_score = np.clip(previous_score, 25, 65)
        
        record = {
            'Hours_Studied': np.random.randint(5, 15),
            'Attendance': np.random.randint(50, 75),
            'Parental_Involvement': np.random.choice(['Low', 'Medium'], p=[0.6, 0.4]),
            'Access_to_Resources': np.random.choice(['Low', 'Medium'], p=[0.65, 0.35]),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], p=[0.35, 0.65]),
            'Sleep_Hours': np.random.randint(5, 8),
            'Previous_Scores': previous_score,
            'Motivation_Level': np.random.choice(['Low', 'Medium'], p=[0.5, 0.5]),
            'Internet_Access': np.random.choice(['Yes', 'No'], p=[0.7, 0.3]),
            'Tutoring_Sessions': np.random.randint(0, 4),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], p=[0.45, 0.45, 0.1]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
            'School_Type': np.random.choice(['Public', 'Private'], p=[0.75, 0.25]),
            'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.25, 0.5, 0.25]),
            'Physical_Activity': np.random.randint(1, 4),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], p=[0.2, 0.8]),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], p=[0.5, 0.35, 0.15]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], p=[0.3, 0.4, 0.3]),
            'Gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'Exam_Score': base_score
        }
        synthetic_records.append(record)
    
    # Pass grade students (40-44 exam scores)
    print(f"Creating {n_pass} Pass grade records (40-44%)...")
    for _ in range(n_pass):
        base_score = np.random.randint(40, 45)
        previous_score = np.random.normal(base_score - 8, 10)
        previous_score = np.clip(previous_score, 15, 55)
        
        record = {
            'Hours_Studied': np.random.randint(2, 10),
            'Attendance': np.random.randint(30, 60),
            'Parental_Involvement': np.random.choice(['Low', 'Medium'], p=[0.75, 0.25]),
            'Access_to_Resources': np.random.choice(['Low', 'Medium'], p=[0.8, 0.2]),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], p=[0.25, 0.75]),
            'Sleep_Hours': np.random.randint(4, 8),
            'Previous_Scores': previous_score,
            'Motivation_Level': np.random.choice(['Low', 'Medium'], p=[0.7, 0.3]),
            'Internet_Access': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
            'Tutoring_Sessions': np.random.randint(0, 3),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], p=[0.45, 0.45, 0.1]),
            'School_Type': np.random.choice(['Public', 'Private'], p=[0.85, 0.15]),
            'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.2, 0.4, 0.4]),
            'Physical_Activity': np.random.randint(0, 3),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], p=[0.7, 0.25, 0.05]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], p=[0.25, 0.35, 0.4]),
            'Gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'Exam_Score': base_score
        }
        synthetic_records.append(record)
    
    # Fail grade students (0-39 exam scores)
    print(f"Creating {n_fail} Fail grade records (0-39%)...")
    for _ in range(n_fail):
        base_score = np.random.randint(0, 40)
        previous_score = np.random.normal(base_score - 5, 12)
        previous_score = np.clip(previous_score, 0, 50)
        
        record = {
            'Hours_Studied': np.random.randint(0, 8),
            'Attendance': np.random.randint(10, 50),
            'Parental_Involvement': np.random.choice(['Low', 'Medium'], p=[0.85, 0.15]),
            'Access_to_Resources': np.random.choice(['Low', 'Medium'], p=[0.85, 0.15]),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'Sleep_Hours': np.random.randint(3, 7),
            'Previous_Scores': previous_score,
            'Motivation_Level': np.random.choice(['Low', 'Medium'], p=[0.9, 0.1]),
            'Internet_Access': np.random.choice(['Yes', 'No'], p=[0.5, 0.5]),
            'Tutoring_Sessions': np.random.randint(0, 2),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], p=[0.75, 0.22, 0.03]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], p=[0.55, 0.35, 0.1]),
            'School_Type': np.random.choice(['Public', 'Private'], p=[0.9, 0.1]),
            'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.1, 0.3, 0.6]),
            'Physical_Activity': np.random.randint(0, 2),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], p=[0.4, 0.6]),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], p=[0.8, 0.17, 0.03]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], p=[0.2, 0.3, 0.5]),
            'Gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'Exam_Score': base_score
        }
        synthetic_records.append(record)
    
    # Create DataFrame from synthetic records
    synthetic_df = pd.DataFrame(synthetic_records)
    
    # Combine with original data
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    return augmented_df, synthetic_df

def create_high_performers_synthetic(original_df, n_first_class=200, n_second_upper=300):
    """
    Create synthetic high-performing students with logical consistency
    """
    synthetic_records = []
    
    # First Class students (70-100 exam scores)
    print(f"Creating {n_first_class} First Class records (70-100%)...")
    for _ in range(n_first_class):
        base_score = np.random.randint(70, 95)
        previous_score = np.random.normal(base_score + 2, 8)
        previous_score = np.clip(previous_score, 65, 100)
        
        record = {
            'Hours_Studied': np.random.randint(15, 35),
            'Attendance': np.random.randint(80, 100),
            'Parental_Involvement': np.random.choice(['Medium', 'High'], p=[0.3, 0.7]),
            'Access_to_Resources': np.random.choice(['Medium', 'High'], p=[0.2, 0.8]),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], p=[0.7, 0.3]),
            'Sleep_Hours': np.random.randint(6, 9),
            'Previous_Scores': previous_score,
            'Motivation_Level': np.random.choice(['Medium', 'High'], p=[0.2, 0.8]),
            'Internet_Access': np.random.choice(['Yes', 'No'], p=[0.9, 0.1]),
            'Tutoring_Sessions': np.random.randint(2, 8),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.4, 0.5]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.3, 0.6]),
            'School_Type': np.random.choice(['Public', 'Private'], p=[0.4, 0.6]),
            'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.7, 0.25, 0.05]),
            'Physical_Activity': np.random.randint(2, 5),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], p=[0.05, 0.95]),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], p=[0.15, 0.4, 0.45]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], p=[0.5, 0.35, 0.15]),
            'Gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'Exam_Score': base_score
        }
        synthetic_records.append(record)
    
    # Second Class Upper students (60-69 exam scores)
    print(f"Creating {n_second_upper} Second Class Upper records (60-69%)...")
    for _ in range(n_second_upper):
        base_score = np.random.randint(60, 70)
        previous_score = np.random.normal(base_score, 10)
        previous_score = np.clip(previous_score, 50, 85)
        
        record = {
            'Hours_Studied': np.random.randint(12, 25),
            'Attendance': np.random.randint(70, 90),
            'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]),
            'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
            'Sleep_Hours': np.random.randint(6, 9),
            'Previous_Scores': previous_score,
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], p=[0.1, 0.4, 0.5]),
            'Internet_Access': np.random.choice(['Yes', 'No'], p=[0.85, 0.15]),
            'Tutoring_Sessions': np.random.randint(1, 6),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], p=[0.15, 0.45, 0.4]),
            'School_Type': np.random.choice(['Public', 'Private'], p=[0.6, 0.4]),
            'Peer_Influence': np.random.choice(['Positive', 'Neutral', 'Negative'], p=[0.6, 0.3, 0.1]),
            'Physical_Activity': np.random.randint(1, 4),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], p=[0.1, 0.9]),
            'Parental_Education_Level': np.random.choice(['High School', 'College', 'Postgraduate'], p=[0.3, 0.45, 0.25]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], p=[0.4, 0.4, 0.2]),
            'Gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
            'Exam_Score': base_score
        }
        synthetic_records.append(record)
    
    return pd.DataFrame(synthetic_records)

def validate_comprehensive_grade_distribution(df):
    """
    Validate grade distribution for comprehensive model training
    """
    df['Grade_Class'] = df['Exam_Score'].apply(enhanced_classify_grade)
    grade_counts = df['Grade_Class'].value_counts()
    
    print("\n📊 Grade Distribution Analysis:")
    print("-" * 40)
    for grade in ['First Class', 'Second Class Upper', 'Second Class Lower', 'Third Class', 'Pass', 'Fail']:
        count = grade_counts.get(grade, 0)
        percentage = (count / len(df)) * 100
        print(f"{grade:<20}: {count:>4} ({percentage:>5.1f}%)")
    
    # Check for missing grades
    expected_grades = ['First Class', 'Second Class Upper', 'Second Class Lower', 'Third Class', 'Pass', 'Fail']
    missing_grades = [grade for grade in expected_grades if grade not in grade_counts.index]
    
    if missing_grades:
        print(f"\n⚠️  Missing grades: {', '.join(missing_grades)}")
        return False
    else:
        print(f"\n✅ All grade categories present!")
        return True

def augment_dataset_comprehensive(original_csv_path="data.csv", 
                                output_csv_path="comprehensive_augmented_data.csv",
                                n_third_class=500, n_pass=400, n_fail=300):
    """
    Complete pipeline for creating comprehensive augmented dataset for app.py
    """
    print("🎓 COMPREHENSIVE GRADE PREDICTION DATA AUGMENTATION")
    print("="*60)
    
    # Load original data
    try:
        original_df = pd.read_csv(original_csv_path)
        print(f"✅ Original data loaded: {len(original_df)} records")
    except FileNotFoundError:
        print(f"❌ Error: {original_csv_path} not found!")
        return None
    
    # Validate original data has required columns
    required_columns = ['Hours_Studied', 'Previous_Scores', 'Attendance', 'Exam_Score']
    missing_columns = [col for col in required_columns if col not in original_df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        return None
    
    # Create low-performing synthetic data
    print("\n🔧 Creating underrepresented grade categories...")
    augmented_df, low_performers = create_logically_consistent_dataset(
        original_df, n_third_class=n_third_class, n_pass=n_pass, n_fail=n_fail
    )
    
    # Create high-performing synthetic data if needed
    print("\n🚀 Creating high-performing students...")
    high_performers = create_high_performers_synthetic(
        original_df, n_first_class=150, n_second_upper=200
    )
    
    # Combine all data
    final_df = pd.concat([augmented_df, high_performers], ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Validate grade distribution
    is_balanced = validate_comprehensive_grade_distribution(final_df)
    
    # Add feature engineering (matching app.py)
    final_df['Study_Attendance_Interaction'] = final_df['Hours_Studied'] * final_df['Attendance']
    final_df['Previous_Scores_Binned'] = pd.cut(
        final_df['Previous_Scores'], 
        bins=[0, 60, 80, 100], 
        labels=['Low', 'Medium', 'High']
    ).astype(str)
    
    # Save the comprehensive dataset
    final_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Comprehensive dataset saved: {output_csv_path}")
    print(f"📈 Total records: {len(original_df)} → {len(final_df)} (+{len(final_df) - len(original_df)})")
    
    # Validate correlation
    correlation = final_df['Previous_Scores'].corr(final_df['Exam_Score'])
    print(f"📊 Previous_Scores ↔ Exam_Score correlation: {correlation:.3f}")
    
    if correlation > 0.5:
        print("✅ Strong logical consistency achieved!")
    else:
        print("⚠️  Consider adjusting synthetic data generation")
    
    return final_df

def retrain_with_comprehensive_data(data_csv_path="comprehensive_augmented_data.csv", 
                                  model_output_path="comprehensive_balanced_grade_model.pkl"):
    """
    Retrain model with comprehensive augmented data - matching app.py structure
    """
    print("\n🤖 RETRAINING MODEL WITH COMPREHENSIVE DATA")
    print("="*50)
    
    # Load the augmented data
    try:
        df = pd.read_csv(data_csv_path)
        print(f"✅ Loaded augmented data: {len(df)} records")
    except FileNotFoundError:
        print(f"❌ Error: {data_csv_path} not found!")
        return None
    
    # Create target variable
    df['Grade_Class'] = df['Exam_Score'].apply(enhanced_classify_grade)
    
    # Prepare features (matching your app.py expected features)
    feature_columns = [
        'Hours_Studied', 'Previous_Scores', 'Attendance', 'Tutoring_Sessions',
        'Sleep_Hours', 'Physical_Activity', 'School_Type', 'Teacher_Quality',
        'Gender', 'Internet_Access', 'Learning_Disabilities', 'Distance_from_Home',
        'Parental_Involvement', 'Parental_Education_Level', 'Family_Income',
        'Access_to_Resources', 'Motivation_Level', 'Peer_Influence',
        'Extracurricular_Activities', 'Study_Attendance_Interaction', 'Previous_Scores_Binned'
    ]
    
    # Check for missing features
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return None
    
    X = df[feature_columns]
    y = df['Grade_Class']
    
    # Identify categorical and numerical columns
    categorical_features = [
        'School_Type', 'Teacher_Quality', 'Gender', 'Internet_Access',
        'Learning_Disabilities', 'Distance_from_Home', 'Parental_Involvement',
        'Parental_Education_Level', 'Family_Income', 'Access_to_Resources',
        'Motivation_Level', 'Peer_Influence', 'Extracurricular_Activities',
        'Previous_Scores_Binned'
    ]
    
    numerical_features = [
        'Hours_Studied', 'Previous_Scores', 'Attendance', 'Tutoring_Sessions',
        'Sleep_Hours', 'Physical_Activity', 'Study_Attendance_Interaction'
    ]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("🏋️  Training comprehensive model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"📊 Training Accuracy: {train_score:.4f}")
    print(f"📊 Testing Accuracy: {test_score:.4f}")
    
    # Detailed classification report
    y_pred = pipeline.predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(pipeline, model_output_path)
    print(f"✅ Model saved: {model_output_path}")
    
    return pipeline

# Main execution for testing
if __name__ == "__main__":
    # Create comprehensive augmented dataset
    print("Step 1: Creating comprehensive augmented dataset...")
    augmented_df = augment_dataset_comprehensive(
        original_csv_path="data.csv",
        output_csv_path="comprehensive_augmented_data.csv",
        n_third_class=500,
        n_pass=400,
        n_fail=300
    )
    
    if augmented_df is not None:
        # Retrain model with comprehensive data
        print("\nStep 2: Retraining model...")
        model = retrain_with_comprehensive_data(
            "comprehensive_augmented_data.csv",
            "comprehensive_balanced_grade_model.pkl"
        )
        
        if model is not None:
            print("\n🎉 SUCCESS! Your app.py can now use:")
            print("   📄 Dataset: comprehensive_augmented_data.csv")
            print("   🤖 Model: comprehensive_balanced_grade_model.pkl")
            print("\n🚀 The model will be automatically loaded in your app!")
    else:
        print("❌ Failed to create augmented dataset. Check your data.csv file.")