import mysql.connector
from mysql.connector import Error
import streamlit as st
import bcrypt

def connect_to_db():
    """Connect to cloud MySQL database with fallback to local"""
    try:
        # Try to get credentials from Streamlit secrets (for cloud deployment)
        if hasattr(st, 'secrets') and 'database' in st.secrets:
            connection = mysql.connector.connect(
                host=st.secrets.database.host,
                user=st.secrets.database.user,
                password=st.secrets.database.password,
                database=st.secrets.database.database,
                port=st.secrets.database.port,
                ssl_disabled=False,
                autocommit=True
            )
        else:
            # Fallback for local development
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="pass123",
                database="grade_prediction",
                autocommit=True
            )
        
        if connection.is_connected():
            return connection
            
    except Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Authentication Functions
def add_user(email, password, role):
    """Add a new user to the database"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, %s)", (email, hashed_password, role))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.errors.IntegrityError:
        st.error("User with this email already exists.")
        return False
    except Error as e:
        st.error(f"Error adding user: {e}")
        return False
    finally:
        if conn.is_connected():
            conn.close()

def authenticate_user(email, password):
    """Authenticate user login"""
    conn = connect_to_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT email, password, role FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            return {"email": user[0], "role": user[2]}
        return None
    except Error as e:
        st.error(f"Authentication error: {e}")
        return None

def initialize_database():
    """Initialize all database tables"""
    conn = connect_to_db()
    if conn is None:
        st.error("Failed to connect to database")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(50)
            )
        """)
        
        # Create reset tokens table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reset_tokens (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL,
                token VARCHAR(10) NOT NULL,
                expires_at DATETIME NOT NULL,
                used BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address VARCHAR(45),
                user_agent TEXT,
                INDEX idx_email (email),
                INDEX idx_token (token),
                INDEX idx_expires (expires_at)
            )
        """)
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                input_features TEXT,
                predicted_class VARCHAR(50),
                prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create feedback table (optional)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                comment TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create students table (for grade prediction)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create grade predictions table with all features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grade_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT,
                hours_studied FLOAT,
                attendance FLOAT,
                parental_involvement VARCHAR(50),
                access_to_resources VARCHAR(50),
                extracurricular_activities VARCHAR(50),
                sleep_hours FLOAT,
                previous_scores FLOAT,
                motivation_level VARCHAR(50),
                internet_access VARCHAR(50),
                tutoring_sessions INT,
                family_income VARCHAR(50),
                teacher_quality VARCHAR(50),
                school_type VARCHAR(50),
                peer_influence VARCHAR(50),
                physical_activity VARCHAR(50),
                learning_disabilities VARCHAR(50),
                parental_education_level VARCHAR(50),
                distance_from_home VARCHAR(50),
                gender VARCHAR(50),
                predicted_exam_score FLOAT,
                actual_exam_score FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        st.error(f"Database initialization error: {e}")
        return False

# Grade Prediction Functions
def save_student_data(name, email):
    """Save student data to database"""
    conn = connect_to_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO students (name, email) VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE name = %s
        """, (name, email, name))
        
        # Get the student ID
        cursor.execute("SELECT id FROM students WHERE email = %s", (email,))
        student_id = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        return student_id
        
    except Error as e:
        st.error(f"Error saving student data: {e}")
        return None

def save_grade_prediction(student_id, features_dict):
    """Save grade prediction to database with all features"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO grade_predictions 
            (student_id, hours_studied, attendance, parental_involvement, access_to_resources,
             extracurricular_activities, sleep_hours, previous_scores, motivation_level,
             internet_access, tutoring_sessions, family_income, teacher_quality, school_type,
             peer_influence, physical_activity, learning_disabilities, parental_education_level,
             distance_from_home, gender, predicted_exam_score, actual_exam_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            student_id,
            features_dict.get('hours_studied', 0),
            features_dict.get('attendance', 0),
            features_dict.get('parental_involvement', ''),
            features_dict.get('access_to_resources', ''),
            features_dict.get('extracurricular_activities', ''),
            features_dict.get('sleep_hours', 0),
            features_dict.get('previous_scores', 0),
            features_dict.get('motivation_level', ''),
            features_dict.get('internet_access', ''),
            features_dict.get('tutoring_sessions', 0),
            features_dict.get('family_income', ''),
            features_dict.get('teacher_quality', ''),
            features_dict.get('school_type', ''),
            features_dict.get('peer_influence', ''),
            features_dict.get('physical_activity', ''),
            features_dict.get('learning_disabilities', ''),
            features_dict.get('parental_education_level', ''),
            features_dict.get('distance_from_home', ''),
            features_dict.get('gender', ''),
            features_dict.get('predicted_exam_score', 0),
            features_dict.get('actual_exam_score', None)
        ))
        
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        st.error(f"Error saving prediction: {e}")
        return False

def get_student_history(student_id):
    """Get student's prediction history with all features"""
    conn = connect_to_db()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hours_studied, attendance, parental_involvement, access_to_resources,
                   extracurricular_activities, sleep_hours, previous_scores, motivation_level,
                   internet_access, tutoring_sessions, family_income, teacher_quality, school_type,
                   peer_influence, physical_activity, learning_disabilities, parental_education_level,
                   distance_from_home, gender, predicted_exam_score, actual_exam_score, created_at
            FROM grade_predictions 
            WHERE student_id = %s 
            ORDER BY created_at DESC
        """, (student_id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
        
    except Error as e:
        st.error(f"Error fetching student history: {e}")
        return []

def save_prediction(user_id, input_features, predicted_class):
    """Save general prediction to database (for compatibility)"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (user_id, input_features, predicted_class)
            VALUES (%s, %s, %s)
        """, (user_id, input_features, predicted_class))
        
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        st.error(f"Error saving prediction: {e}")
        return False

def save_feedback(user_id, comment):
    """Save user feedback"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_id, comment)
            VALUES (%s, %s)
        """, (user_id, comment))
        
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        st.error(f"Error saving feedback: {e}")
        return False

def get_user_id_by_email(email):
    """Get user ID by email"""
    conn = connect_to_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] if result else None
        
    except Error as e:
        st.error(f"Error getting user ID: {e}")
        return None

def get_all_predictions():
    """Get all predictions for analysis with all features"""
    conn = connect_to_db()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT gp.id, gp.student_id, s.name, s.email,
                   gp.hours_studied, gp.attendance, gp.parental_involvement, gp.access_to_resources,
                   gp.extracurricular_activities, gp.sleep_hours, gp.previous_scores, gp.motivation_level,
                   gp.internet_access, gp.tutoring_sessions, gp.family_income, gp.teacher_quality, 
                   gp.school_type, gp.peer_influence, gp.physical_activity, gp.learning_disabilities, 
                   gp.parental_education_level, gp.distance_from_home, gp.gender, 
                   gp.predicted_exam_score, gp.actual_exam_score, gp.created_at
            FROM grade_predictions gp
            JOIN students s ON gp.student_id = s.id
            ORDER BY gp.created_at DESC
        """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
        
    except Error as e:
        st.error(f"Error fetching all predictions: {e}")
        return []

def get_prediction_column_names():
    """Get all column names for predictions in order"""
    return [
        'id', 'student_id', 'name', 'email',
        'hours_studied', 'attendance', 'parental_involvement', 'access_to_resources',
        'extracurricular_activities', 'sleep_hours', 'previous_scores', 'motivation_level',
        'internet_access', 'tutoring_sessions', 'family_income', 'teacher_quality',
        'school_type', 'peer_influence', 'physical_activity', 'learning_disabilities',
        'parental_education_level', 'distance_from_home', 'gender',
        'predicted_exam_score', 'actual_exam_score', 'created_at'
    ]

def get_student_history_with_columns(student_id):
    """Get student's prediction history with column names"""
    conn = connect_to_db()
    if conn is None:
        return [], []
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, hours_studied, attendance, parental_involvement, access_to_resources,
                   extracurricular_activities, sleep_hours, previous_scores, motivation_level,
                   internet_access, tutoring_sessions, family_income, teacher_quality, school_type,
                   peer_influence, physical_activity, learning_disabilities, parental_education_level,
                   distance_from_home, gender, predicted_exam_score, actual_exam_score, created_at
            FROM grade_predictions 
            WHERE student_id = %s 
            ORDER BY created_at DESC
        """, (student_id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        columns = [
            'id', 'hours_studied', 'attendance', 'parental_involvement', 'access_to_resources',
            'extracurricular_activities', 'sleep_hours', 'previous_scores', 'motivation_level',
            'internet_access', 'tutoring_sessions', 'family_income', 'teacher_quality', 'school_type',
            'peer_influence', 'physical_activity', 'learning_disabilities', 'parental_education_level',
            'distance_from_home', 'gender', 'predicted_exam_score', 'actual_exam_score', 'created_at'
        ]
        
        return results, columns
        
    except Error as e:
        st.error(f"Error fetching student history: {e}")
        return [], []

def get_prediction_statistics():
    """Get comprehensive statistics about predictions"""
    conn = connect_to_db()
    if conn is None:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(hours_studied) as avg_hours_studied,
                AVG(attendance) as avg_attendance,
                AVG(sleep_hours) as avg_sleep_hours,
                AVG(previous_scores) as avg_previous_scores,
                AVG(tutoring_sessions) as avg_tutoring_sessions,
                AVG(predicted_exam_score) as avg_predicted_score,
                AVG(actual_exam_score) as avg_actual_score,
                MIN(predicted_exam_score) as min_predicted,
                MAX(predicted_exam_score) as max_predicted,
                MIN(actual_exam_score) as min_actual,
                MAX(actual_exam_score) as max_actual,
                COUNT(CASE WHEN actual_exam_score IS NOT NULL THEN 1 END) as predictions_with_actual_scores
            FROM grade_predictions
        """)
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
        
    except Error as e:
        st.error(f"Error fetching statistics: {e}")
        return None

def update_actual_exam_score(prediction_id, actual_score):
    """Update actual exam score for a prediction"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE grade_predictions 
            SET actual_exam_score = %s 
            WHERE id = %s
        """, (actual_score, prediction_id))
        
        cursor.close()
        conn.close()
        return True
        
    except Error as e:
        st.error(f"Error updating actual score: {e}")
        return False

def get_predictions_by_feature(feature_name, feature_value):
    """Get predictions filtered by a specific feature value"""
    conn = connect_to_db()
    if conn is None:
        return []
    
    try:
        cursor = conn.cursor()
        query = f"""
            SELECT gp.id, gp.student_id, s.name, s.email,
                   gp.hours_studied, gp.attendance, gp.parental_involvement, gp.access_to_resources,
                   gp.extracurricular_activities, gp.sleep_hours, gp.previous_scores, gp.motivation_level,
                   gp.internet_access, gp.tutoring_sessions, gp.family_income, gp.teacher_quality, 
                   gp.school_type, gp.peer_influence, gp.physical_activity, gp.learning_disabilities, 
                   gp.parental_education_level, gp.distance_from_home, gp.gender, 
                   gp.predicted_exam_score, gp.actual_exam_score, gp.created_at
            FROM grade_predictions gp
            JOIN students s ON gp.student_id = s.id
            WHERE gp.{feature_name} = %s
            ORDER BY gp.created_at DESC
        """
        
        cursor.execute(query, (feature_value,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
        
    except Error as e:
        st.error(f"Error fetching predictions by feature: {e}")
        return []

# Legacy Functions for Backward Compatibility
def create_users_table():
    """Legacy function - use initialize_database() instead"""
    return initialize_database()

def create_reset_tokens_table():
    """Legacy function - use initialize_database() instead"""
    return initialize_database()

def test_database_connection():
    """Test database connection"""
    conn = connect_to_db()
    if conn:
        st.success("✅ Database connected successfully!")
        conn.close()
        return True
    else:
        st.error("❌ Database connection failed!")
        return False