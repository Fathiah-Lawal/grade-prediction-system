import streamlit as st
import bcrypt
import mysql.connector
import smtplib
import random
import string
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mysql.connector import Error
from db import connect_to_db  # Import your existing connection function

# Email configuration - Now using environment variables
EMAIL_CONFIG = {
    'smtp_server': st.secrets["email"]["smtp_server"],
    'smtp_port': int(st.secrets["email"]["smtp_port"]),
    'sender_email': st.secrets["email"]["sender_email"],
    'sender_password': st.secrets["email"]["sender_password"],
    'sender_name': st.secrets["email"]["sender_name"]
}

# Validate email configuration
def validate_email_config():
    """Validate that email configuration is properly set"""
    if not EMAIL_CONFIG['sender_email']:
        raise ValueError("SENDER_EMAIL environment variable is not set")
    if not EMAIL_CONFIG['sender_password']:
        raise ValueError("SENDER_PASSWORD environment variable is not set")
    return True
def generate_reset_token(length=6):
    """Generate a random reset token"""
    return ''.join(random.choices(string.digits, k=length))

def send_reset_email(email, token):
    """Send password reset email with token"""
    try:
        # Validate email configuration first
        validate_email_config()

        # Create message
        message = MIMEMultipart()
        message["From"] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        message["To"] = email
        message["Subject"] = "EduPredict - Password Reset Code"
        
        # Create email body
        body = f"""
        Hi there,
        
        You requested a password reset for your EduPredict account.
        
        Your password reset code is: {token}
        
        This code will expire in 15 minutes for security reasons.
        
        If you didn't request this reset, please ignore this email.
        
        Best regards,
        EduPredict Team
        """
        
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        server.send_message(message)
        server.quit()
        
        return True
    except ValueError as e:
        st.error(f"Email configuration error: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def store_reset_token(email, token, ip_address="unknown", user_agent="unknown"):
    """Store reset token in database"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(minutes=15)  # Token expires in 15 minutes
        
        cursor.execute("""
            INSERT INTO reset_tokens (email, token, expires_at, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s)
        """, (email, token, expires_at, ip_address, user_agent))
        
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error storing reset token: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_reset_token(email, token):
    """Verify if reset token is valid and not expired"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, expires_at FROM reset_tokens 
            WHERE email = %s AND token = %s AND used = FALSE
            ORDER BY created_at DESC LIMIT 1
        """, (email, token))
        
        result = cursor.fetchone()
        
        if result:
            token_id, expires_at = result
            if datetime.now() < expires_at:
                return token_id
            else:
                # Token expired, mark as used
                cursor.execute("UPDATE reset_tokens SET used = TRUE WHERE id = %s", (token_id,))
                conn.commit()
                return False
        return False
    except Error as e:
        st.error(f"Error verifying token: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def mark_token_as_used(token_id):
    """Mark token as used after successful password reset"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE reset_tokens SET used = TRUE WHERE id = %s", (token_id,))
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error marking token as used: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def check_user_exists(email):
    """Check if user exists in database"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        return result is not None
    except Error as e:
        st.error(f"Error checking user: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def update_user_password(email, new_password):
    """Update user's password in database"""
    conn = connect_to_db()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, email))
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        st.error(f"Error updating password: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def cleanup_expired_tokens():
    """Clean up expired tokens from database"""
    conn = connect_to_db()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reset_tokens WHERE expires_at < %s", (datetime.now(),))
        conn.commit()
    except Error as e:
        st.error(f"Error cleaning up tokens: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_rate_limit_info(email):
    """Check rate limiting for password reset requests"""
    conn = connect_to_db()
    if conn is None:
        return True, 0  # Allow if DB connection fails
    
    try:
        cursor = conn.cursor()
        # Check requests in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute("""
            SELECT COUNT(*) FROM reset_tokens 
            WHERE email = %s AND created_at > %s
        """, (email, one_hour_ago))
        
        count = cursor.fetchone()[0]
        max_requests = 3  # Maximum 3 requests per hour
        
        return count < max_requests, count
    except Error as e:
        st.error(f"Error checking rate limit: {e}")
        return True, 0
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def forgot_password_ui():
    """Main UI for forgot password functionality"""
    st.title("🔐 Reset Your Password")
    
    # Initialize session state
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = 'request'
    if 'reset_email' not in st.session_state:
        st.session_state.reset_email = ''
    
    # Clean up expired tokens
    cleanup_expired_tokens()
    
    if st.session_state.reset_step == 'request':
        st.markdown("### Step 1: Enter Your Email")
        st.info("Enter your registered email address to receive a password reset code.")
        
        email = st.text_input("Email Address", placeholder="your-email@example.com")
        
        if st.button("Send Reset Code", key="send_reset_code"):
            if not email:
                st.error("Please enter your email address.")
                return
            
            if not email.count('@') == 1 or not '.' in email.split('@')[1]:
                st.error("Please enter a valid email address.")
                return
            
            # Check if user exists
            if not check_user_exists(email):
                st.error("No account found with this email address.")
                return
            
            # Check rate limiting
            can_request, request_count = get_rate_limit_info(email)
            if not can_request:
                st.error(f"Too many reset requests. You've made {request_count} requests in the last hour. Please wait before trying again.")
                return
            
            # Generate and store token
            token = generate_reset_token()
            
            # Get client info (simplified for Streamlit)
            ip_address = "streamlit_app"
            user_agent = "streamlit_browser"
            
            if store_reset_token(email, token, ip_address, user_agent):
                if send_reset_email(email, token):
                    st.session_state.reset_email = email
                    st.session_state.reset_step = 'verify'
                    st.success("Reset code sent to your email!")
                    st.rerun()
                else:
                    st.error("Failed to send reset code. Please try again.")
            else:
                st.error("Failed to generate reset code. Please try again.")

                        
           
    elif st.session_state.reset_step == 'verify':
        st.markdown("### Step 2: Enter Reset Code")
        st.info(f"Enter the 6-digit code sent to {st.session_state.reset_email}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            reset_code = st.text_input("Reset Code", placeholder="123456", max_chars=6)
        with col2:
            if st.button("Resend Code", key="resend_code"):
                # Check rate limiting again
                can_request, request_count = get_rate_limit_info(st.session_state.reset_email)
                if not can_request:
                    st.error(f"Too many requests. Please wait before requesting another code.")
                else:
                    # Generate new token
                    token = generate_reset_token()
                    if store_reset_token(st.session_state.reset_email, token):
                        st.success(f"New reset code: **{token}**")
                    else:
                        st.error("Failed to generate new code.")
        
        if st.button("Verify Code", key="verify_code"):
            if not reset_code:
                st.error("Please enter the reset code.")
                return
            
            if len(reset_code) != 6 or not reset_code.isdigit():
                st.error("Reset code must be 6 digits.")
                return
            
            token_id = verify_reset_token(st.session_state.reset_email, reset_code)
            if token_id:
                st.session_state.reset_step = 'reset'
                st.session_state.token_id = token_id
                st.success("Code verified! Now set your new password.")
                st.rerun()
            else:
                st.error("Invalid or expired reset code. Please try again.")
        
        if st.button("← Back to Email Entry", key="back_to_email"):
            st.session_state.reset_step = 'request'
            st.session_state.reset_email = ''
            st.rerun()
    
    elif st.session_state.reset_step == 'reset':
        st.markdown("### Step 3: Set New Password")
        st.info("Enter your new password below.")
        
        new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
        
        # Password strength indicator
        if new_password:
            strength_score = 0
            feedback = []
            
            if len(new_password) >= 8:
                strength_score += 1
            else:
                feedback.append("At least 8 characters")
            
            if any(c.isupper() for c in new_password):
                strength_score += 1
            else:
                feedback.append("At least one uppercase letter")
            
            if any(c.islower() for c in new_password):
                strength_score += 1
            else:
                feedback.append("At least one lowercase letter")
            
            if any(c.isdigit() for c in new_password):
                strength_score += 1
            else:
                feedback.append("At least one number")
            
            if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in new_password):
                strength_score += 1
            else:
                feedback.append("At least one special character")
            
            # Show strength
            if strength_score <= 2:
                st.error(f"Password strength: Weak. Missing: {', '.join(feedback)}")
            elif strength_score <= 3:
                st.warning(f"Password strength: Medium. Missing: {', '.join(feedback)}")
            elif strength_score <= 4:
                st.info(f"Password strength: Good. Missing: {', '.join(feedback)}")
            else:
                st.success("Password strength: Strong!")
        
        if st.button("Reset Password", key="reset_password_btn"):
            if not new_password or not confirm_password:
                st.error("Please fill in both password fields.")
                return
            
            if new_password != confirm_password:
                st.error("Passwords do not match.")
                return
            
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
                return
            
            # Update password
            if update_user_password(st.session_state.reset_email, new_password):
                # Mark token as used
                mark_token_as_used(st.session_state.token_id)
                
                st.success("🎉 Password reset successful!")
                st.info("You can now log in with your new password.")
                
                # Clear session state
                st.session_state.reset_step = 'request'
                st.session_state.reset_email = ''
                if 'token_id' in st.session_state:
                    del st.session_state.token_id
                
                # Add button to go back to login
                if st.button("← Back to Login", key="back_to_login"):
                    st.session_state.reset_step = 'request'
                    st.rerun()
            else:
                st.error("Failed to reset password. Please try again.")

# Function to integrate with your existing auth system
def add_forgot_password_to_auth():
    """Add forgot password option to existing auth"""
    if st.button("Forgot Password?", key="forgot_password_link"):
        st.session_state.show_forgot_password = True
        st.rerun()

# Modified auth function that includes forgot password
def enhanced_auth_app():
    """Enhanced authentication with forgot password"""
    st.title("EduPredict - Authentication")
    
    # Check if showing forgot password
    if st.session_state.get('show_forgot_password', False):
        forgot_password_ui()
        
        # Add back to login button
        if st.button("← Back to Login", key="back_to_main_login"):
            st.session_state.show_forgot_password = False
            st.rerun()
        return
    
    # Your existing auth code here
    action = st.selectbox("Choose Action", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if action == "Sign Up":
        role = st.radio("Select Role", ["Student", "Educator"])
        if st.button("Sign Up"):
            if email and password:
                # Your existing signup logic
                from db import add_user
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
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Login"):
                if email and password:
                    # Your existing login logic
                    from db import authenticate_user
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
        
        with col2:
            if st.button("Forgot Password?"):
                st.session_state.show_forgot_password = True
                st.rerun()

if __name__ == "__main__":
    # For testing the module independently
    forgot_password_ui()