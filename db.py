import streamlit as st
import bcrypt
import mysql.connector

# DB connection
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="pass123",
        database="grade_prediction"
    )

# Create users table
def create_users_table():
    conn = connect_to_db()
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
    conn.close()

# Create reset tokens table
def create_reset_tokens_table():
    conn = connect_to_db()
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
    conn.close()

# Add user
def add_user(email, password, role):
    conn = connect_to_db()
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        cursor.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, %s)", (email, hashed_password, role))
        conn.commit()
    except mysql.connector.errors.IntegrityError:
        st.error("User with this email already exists.")
        return False
    finally:
        conn.close()
    return True

# Authenticate user
def authenticate_user(email, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT email, password, role FROM users WHERE email=%s", (email,))
    user = cursor.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
        return {"email": user[0], "role": user[2]}
    return None

# Initialize all required tables
def initialize_database():
    conn = connect_to_db()
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
    cursor.close()
    conn.close()