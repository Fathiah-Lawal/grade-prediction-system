import streamlit as st
import bcrypt
import mysql.connector
from mysql.connector import Error

# DB connection - Updated to work with Railway MySQL
def connect_to_db():
    try:
        # Check if running on Streamlit Cloud with secrets
        if hasattr(st.secrets, "database"):
            return mysql.connector.connect(
                host=st.secrets["database"]["host"],
                user=st.secrets["database"]["user"],
                password=st.secrets["database"]["password"],
                database=st.secrets["database"]["database"],
                port=int(st.secrets["database"].get("port", 3306))  # Use .get here
            )
        else:
            # Fallback for local development
            return mysql.connector.connect(
                host="localhost",
                user="root",
                password="pass123",
                database="grade_prediction"
            )
    except mysql.connector.Error as e:
        st.error(f"Database connection failed: {e}")
        return None


# Create users table
def create_users_table():
    conn = connect_to_db()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                role VARCHAR(50)
            )
        """)
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error creating users table: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Create reset tokens table
def create_reset_tokens_table():
    conn = connect_to_db()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
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
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error creating reset tokens table: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Add user
def add_user(email, password, role):
    conn = connect_to_db()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, %s)", (email, hashed_password, role))
        conn.commit()
        return True
    except mysql.connector.errors.IntegrityError:
        st.error("User with this email already exists.")
        return False
    except Error as e:
        st.error(f"Error adding user: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Authenticate user
def authenticate_user(email, password):
    conn = connect_to_db()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT email, password, role FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            return {"email": user[0], "role": user[2]}
        return None
    except Error as e:
        st.error(f"Authentication error: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Initialize all required tables
def initialize_database():
    conn = connect_to_db()
    if conn is None:
        st.error("Failed to connect to database for initialization")
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

        conn.commit()
        return True
    except Error as e:
        st.error(f"Error initializing database: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()