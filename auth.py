import mysql.connector
import hashlib
import streamlit as st
from typing import Optional, Tuple
import re
import bcrypt
from datetime import datetime, timedelta

def connect_to_db():
    """Connect to MySQL database"""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="pass123",
        database="grade_prediction"
    )

def hash_password(password: str) -> str:
    """Hash password using bcrypt (more secure than SHA-256)"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"

def create_users_table():
    """Create users table if it doesn't exist"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT TRUE,
                failed_login_attempts INT DEFAULT 0,
                locked_until DATETIME,
                INDEX idx_email (email),
                INDEX idx_active (is_active),
                INDEX idx_locked (locked_until)
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error creating users table: {str(e)}")
        return False

def register_user(email: str, password: str, first_name: str = "", last_name: str = "") -> Tuple[bool, str]:
    """Register a new user"""
    try:
        # Validate email
        if not validate_email(email):
            return False, "Invalid email format"
        
        # Validate password
        is_valid, message = validate_password(password)
        if not is_valid:
            return False, message
        
        # Create table if not exists
        create_users_table()
        
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT email FROM users WHERE email = %s', (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered"
        
        # Hash password and insert user
        hashed_password = hash_password(password)
        cursor.execute('''
            INSERT INTO users (email, password, first_name, last_name)
            VALUES (%s, %s, %s, %s)
        ''', (email, hashed_password, first_name, last_name))
        
        conn.commit()
        conn.close()
        return True, "Registration successful"
        
    except mysql.connector.Error as e:
        return False, f"Database error: {str(e)}"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(email: str, password: str) -> Tuple[bool, str, dict]:
    """Authenticate user login"""
    try:
        create_users_table()
        
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Get user data
        cursor.execute('''
            SELECT id, email, password, first_name, last_name, is_active, 
                   failed_login_attempts, locked_until
            FROM users WHERE email = %s
        ''', (email,))
        
        user_data = cursor.fetchone()
        
        if not user_data:
            conn.close()
            return False, "Invalid email or password", {}
        
        user_id, user_email, stored_password, first_name, last_name, is_active, failed_attempts, locked_until = user_data
        
        # Check if account is locked
        if locked_until:
            if datetime.now() < locked_until:
                conn.close()
                return False, "Account temporarily locked due to multiple failed attempts", {}
        
        # Check if account is active
        if not is_active:
            conn.close()
            return False, "Account is deactivated", {}
        
        # Verify password
        if not verify_password(password, stored_password):
            # Increment failed attempts
            failed_attempts += 1
            
            # Lock account after 5 failed attempts for 30 minutes
            if failed_attempts >= 5:
                lock_until = datetime.now() + timedelta(minutes=30)
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = %s, locked_until = %s
                    WHERE email = %s
                ''', (failed_attempts, lock_until, email))
            else:
                cursor.execute('''
                    UPDATE users SET failed_login_attempts = %s
                    WHERE email = %s
                ''', (failed_attempts, email))
            
            conn.commit()
            conn.close()
            return False, "Invalid email or password", {}
        
        # Reset failed attempts and update last login
        cursor.execute('''
            UPDATE users SET failed_login_attempts = 0, locked_until = NULL, 
                           last_login = %s
            WHERE email = %s
        ''', (datetime.now(), email))
        
        conn.commit()
        conn.close()
        
        user_info = {
            'id': user_id,
            'email': user_email,
            'first_name': first_name,
            'last_name': last_name
        }
        
        return True, "Login successful", user_info
        
    except mysql.connector.Error as e:
        return False, f"Database error: {str(e)}", {}
    except Exception as e:
        return False, f"Login failed: {str(e)}", {}

def user_exists(email: str) -> bool:
    """Check if user exists in database"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT email FROM users WHERE email = %s', (email,))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    except Exception as e:
        st.error(f"Error checking user existence: {str(e)}")
        return False

def update_user_password(email: str, new_password: str) -> bool:
    """Update user's password in database"""
    try:
        # Validate new password
        is_valid, message = validate_password(new_password)
        if not is_valid:
            st.error(message)
            return False
        
        conn = connect_to_db()
        cursor = conn.cursor()
        
        hashed_password = hash_password(new_password)
        cursor.execute('''
            UPDATE users 
            SET password = %s, failed_login_attempts = 0, locked_until = NULL
            WHERE email = %s
        ''', (hashed_password, email))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating password: {str(e)}")
        return False

def get_user_profile(email: str) -> dict:
    """Get user profile information"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, first_name, last_name, created_at, last_login, is_active
            FROM users WHERE email = %s
        ''', (email,))
        
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return {
                'id': user_data[0],
                'email': user_data[1],
                'first_name': user_data[2],
                'last_name': user_data[3],
                'created_at': user_data[4],
                'last_login': user_data[5],
                'is_active': user_data[6]
            }
        return {}
    except Exception as e:
        st.error(f"Error fetching user profile: {str(e)}")
        return {}

def update_user_profile(email: str, first_name: str, last_name: str) -> bool:
    """Update user profile information"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET first_name = %s, last_name = %s
            WHERE email = %s
        ''', (first_name, last_name, email))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating user profile: {str(e)}")
        return False

def deactivate_user(email: str) -> bool:
    """Deactivate user account"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET is_active = FALSE
            WHERE email = %s
        ''', (email,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deactivating user: {str(e)}")
        return False

def get_user_stats() -> dict:
    """Get user statistics for admin dashboard"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # Active users
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = TRUE')
        active_users = cursor.fetchone()[0]
        
        # Users registered today
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE DATE(created_at) = CURDATE()
        ''')
        today_registrations = cursor.fetchone()[0]
        
        # Users logged in today
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE DATE(last_login) = CURDATE()
        ''')
        today_logins = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'inactive_users': total_users - active_users,
            'today_registrations': today_registrations,
            'today_logins': today_logins
        }
    except Exception as e:
        st.error(f"Error fetching user stats: {str(e)}")
        return {}

def login_interface():
    """Streamlit login interface"""
    st.title("🔐 Login to EduPredict")
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        email = st.text_input("Email Address", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Login", key="login_btn", use_container_width=True):
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    success, message, user_info = authenticate_user(email, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_info
                        st.success(f"Welcome back, {user_info.get('first_name', 'User')}!")
                        st.rerun()
                    else:
                        st.error(message)
        
        with col2:
            if st.button("Forgot Password?", key="forgot_password_btn", use_container_width=True):
                st.session_state.show_forgot_password = True
                st.rerun()
    
    with tab2:
        st.markdown("### Create New Account")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", key="reg_first_name")
        with col2:
            last_name = st.text_input("Last Name", key="reg_last_name")
        
        reg_email = st.text_input("Email Address", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        # Password requirements
        st.markdown("""
        **Password Requirements:**
        - At least 8 characters long
        - Contains uppercase and lowercase letters
        - Contains at least one number
        - Contains at least one special character
        """)
        
        if st.button("Create Account", key="register_btn", use_container_width=True):
            if not all([reg_email, reg_password, confirm_password]):
                st.error("Please fill in all required fields")
            elif reg_password != confirm_password:
                st.error("Passwords don't match")
            else:
                success, message = register_user(reg_email, reg_password, first_name, last_name)
                if success:
                    st.success("Account created successfully! You can now login.")
                else:
                    st.error(message)

def logout():
    """Logout function"""
    if 'logged_in' in st.session_state:
        del st.session_state.logged_in
    if 'user_info' in st.session_state:
        del st.session_state.user_info
    st.success("Logged out successfully!")
    st.rerun()

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('logged_in', False)

def get_current_user():
    """Get current user information"""
    return st.session_state.get('user_info', {})

def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        if not check_authentication():
            st.warning("Please login to access this page")
            login_interface()
            return None
        return func(*args, **kwargs)
    return wrapper

def cleanup_expired_locks():
    """Clean up expired account locks"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET locked_until = NULL, failed_login_attempts = 0
            WHERE locked_until < NOW()
        ''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error cleaning up expired locks: {str(e)}")

# Initialize database on import
try:
    create_users_table()
    cleanup_expired_locks()
except Exception as e:
    st.error(f"Error initializing authentication system: {str(e)}")