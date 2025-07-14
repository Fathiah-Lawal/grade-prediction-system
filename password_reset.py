import smtplib
import mysql.connector
import random
import string
import hashlib
import datetime
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
from typing import Optional, Tuple
from dotenv import load_dotenv
import bcrypt

# Load environment variables
load_dotenv()

# Email configuration from environment variables
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'app_name': os.getenv('APP_NAME', 'EduPredict')
}

# Database configuration (matching your db.py)
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="pass123",
        database="grade_prediction"
    )

def generate_reset_code(length: int = 6) -> str:
    """Generate a random reset code"""
    return ''.join(random.choices(string.digits, k=length))

def create_reset_tokens_table():
    """Create table to store password reset tokens"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute('''
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
    ''')
    
    conn.commit()
    conn.close()

def store_reset_token(email: str, token: str, expires_minutes: int = 15, 
                     ip_address: str = None, user_agent: str = None) -> bool:
    """Store reset token in database with expiration"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Calculate expiration time
        expires_at = datetime.datetime.now() + datetime.timedelta(minutes=expires_minutes)
        
        # Invalidate previous unused tokens for this email
        cursor.execute('''
            UPDATE reset_tokens 
            SET used = TRUE 
            WHERE email = %s AND used = FALSE
        ''', (email,))
        
        # Store the new token
        cursor.execute('''
            INSERT INTO reset_tokens (email, token, expires_at, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s)
        ''', (email, token, expires_at, ip_address, user_agent))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error storing reset token: {str(e)}")
        return False

def verify_reset_token(email: str, token: str) -> bool:
    """Verify if reset token is valid and not expired"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, expires_at, used FROM reset_tokens 
            WHERE email = %s AND token = %s AND used = FALSE
            ORDER BY created_at DESC LIMIT 1
        ''', (email, token))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
        
        token_id, expires_at, used = result
        
        # Check if token is expired
        if datetime.datetime.now() > expires_at:
            return False
        
        return True
    except Exception as e:
        st.error(f"Error verifying reset token: {str(e)}")
        return False

def mark_token_as_used(email: str, token: str) -> bool:
    """Mark reset token as used"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE reset_tokens 
            SET used = TRUE 
            WHERE email = %s AND token = %s
        ''', (email, token))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error marking token as used: {str(e)}")
        return False

def get_reset_attempts(email: str, hours: int = 1) -> int:
    """Get number of reset attempts in the last X hours"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        cursor.execute('''
            SELECT COUNT(*) FROM reset_tokens 
            WHERE email = %s AND created_at > %s
        ''', (email, cutoff_time))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        return 0

def user_exists(email: str) -> bool:
    """Check if user exists in database"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        st.error(f"Error checking user existence: {str(e)}")
        return False

def update_user_password(email: str, new_password: str) -> bool:
    """Update user password in database"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Hash the new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        cursor.execute('''
            UPDATE users 
            SET password = %s 
            WHERE email = %s
        ''', (hashed_password, email))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error updating password: {str(e)}")
        return False

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    special_chars = "!@#$%^&*(),.?\":{}|<>"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password meets all requirements"

def check_email_config() -> bool:
    """Check if email configuration is properly set up"""
    required_configs = ['sender_email', 'sender_password']
    return all(EMAIL_CONFIG.get(key) for key in required_configs)

def send_reset_email(email: str, reset_code: str, user_name: str = "") -> bool:
    """Send password reset email with improved design"""
    try:
        if not check_email_config():
            st.error("Email configuration is not set up properly. Please check your .env file.")
            return False
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = f"{EMAIL_CONFIG['app_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To'] = email
        msg['Subject'] = f"{EMAIL_CONFIG['app_name']} - Password Reset Code"
        
        # HTML email body with better design
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password Reset</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 28px;">🔐 Password Reset</h1>
                <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 16px;">{EMAIL_CONFIG['app_name']}</p>
            </div>
            
            <div style="background: white; padding: 30px; border: 1px solid #ddd; border-top: none;">
                <p style="font-size: 16px; margin-bottom: 20px;">
                    Hello{f" {user_name}" if user_name else ""},
                </p>
                
                <p style="font-size: 16px; margin-bottom: 20px;">
                    You have requested to reset your password for your {EMAIL_CONFIG['app_name']} account.
                </p>
                
                <div style="background: #f8f9fa; border: 2px dashed #667eea; border-radius: 8px; padding: 25px; text-align: center; margin: 25px 0;">
                    <p style="margin: 0 0 10px 0; font-size: 14px; color: #666;">Your password reset code is:</p>
                    <div style="font-size: 32px; font-weight: bold; color: #667eea; letter-spacing: 5px; font-family: 'Courier New', monospace;">
                        {reset_code}
                    </div>
                </div>
                
                <div style="background: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; padding: 15px; margin: 20px 0;">
                    <p style="margin: 0; font-size: 14px; color: #856404;">
                        ⏰ <strong>Important:</strong> This code will expire in 15 minutes for security reasons.
                    </p>
                </div>
                
                <p style="font-size: 16px; margin: 20px 0;">
                    If you didn't request this password reset, please ignore this email or contact our support team if you have concerns about your account security.
                </p>
                
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                
                <p style="font-size: 14px; color: #666; margin: 0;">
                    Best regards,<br>
                    The {EMAIL_CONFIG['app_name']} Team
                </p>
            </div>
            
            <div style="background: #f8f9fa; padding: 20px; border-radius: 0 0 10px 10px; text-align: center;">
                <p style="font-size: 12px; color: #666; margin: 0;">
                    This is an automated message. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Plain text alternative
        text_body = f"""
        Password Reset Request - {EMAIL_CONFIG['app_name']}
        
        Hello{f" {user_name}" if user_name else ""},
        
        You have requested to reset your password for your {EMAIL_CONFIG['app_name']} account.
        
        Your password reset code is: {reset_code}
        
        This code will expire in 15 minutes.
        
        If you didn't request this password reset, please ignore this email.
        
        Best regards,
        The {EMAIL_CONFIG['app_name']} Team
        """
        
        # Attach both versions
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        
        server.sendmail(EMAIL_CONFIG['sender_email'], email, msg.as_string())
        server.quit()
        
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("Email authentication failed. Please check your email credentials.")
        return False
    except smtplib.SMTPException as e:
        st.error(f"Failed to send email: {str(e)}")
        return False
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

def cleanup_expired_tokens():
    """Clean up expired tokens from database"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM reset_tokens 
            WHERE expires_at < NOW() OR created_at < DATE_SUB(NOW(), INTERVAL 7 DAY)
        ''')
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error cleaning up expired tokens: {str(e)}")

def get_user_name(email: str) -> str:
    """Get user's name from database (if exists)"""
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        # Try to get first name if column exists, otherwise return empty string
        cursor.execute("SHOW COLUMNS FROM users LIKE 'first_name'")
        if cursor.fetchone():
            cursor.execute('SELECT first_name FROM users WHERE email = %s', (email,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result and result[0] else ""
        else:
            conn.close()
            return ""
    except Exception as e:
        return ""

def forgot_password_interface():
    """Enhanced Streamlit interface for forgot password functionality"""
    st.title("🔐 Reset Your Password")
    
    # Initialize reset tokens table
    create_reset_tokens_table()
    
    # Clean up expired tokens
    cleanup_expired_tokens()
    
    # Initialize session state
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = 'request'
    if 'reset_email' not in st.session_state:
        st.session_state.reset_email = ''
    
    if st.session_state.reset_step == 'request':
        # Step 1: Request password reset
        st.markdown("### Step 1: Enter your email address")
        
        # Check email configuration
        if not check_email_config():
            st.error("❌ Email service is not configured. Please contact administrator.")
            return 'login'
        
        st.info("💡 We'll send you a secure 6-digit code to verify your identity.")
        
        email = st.text_input("📧 Email Address", placeholder="Enter your registered email address")
        
        if st.button("📤 Send Reset Code", key="send_reset_code", use_container_width=True):
            if not email:
                st.error("Please enter your email address.")
            elif not user_exists(email):
                st.error("No account found with this email address.")
            else:
                # Check rate limiting
                attempts = get_reset_attempts(email, hours=1)
                if attempts >= 3:
                    st.error("Too many reset attempts. Please try again in 1 hour.")
                    return None
                
                # Generate and send reset code
                reset_code = generate_reset_code()
                
                # Get user's name for personalized email
                user_name = get_user_name(email)
                
                with st.spinner("Sending reset code..."):
                    if store_reset_token(email, reset_code) and send_reset_email(email, reset_code, user_name):
                        st.session_state.reset_email = email
                        st.session_state.reset_step = 'verify'
                        st.success("✅ Reset code sent! Check your email inbox.")
                        st.rerun()
                    else:
                        st.error("❌ Failed to send reset code. Please try again.")
        
        # Back to login option
        if st.button("← Back to Login", key="back_to_login"):
            st.session_state.reset_step = 'request'
            st.session_state.reset_email = ''
            return 'login'
    
    elif st.session_state.reset_step == 'verify':
        # Step 2: Verify reset code and set new password
        st.markdown("### Step 2: Enter reset code and new password")
        st.info(f"📧 We sent a 6-digit code to **{st.session_state.reset_email}**")
        
        # Reset code input with resend option
        col1, col2 = st.columns([3, 1])
        
        with col1:
            reset_code = st.text_input("🔢 Reset Code", placeholder="Enter 6-digit code", max_chars=6)
        with col2:
            if st.button("🔄 Resend", key="resend_code"):
                attempts = get_reset_attempts(st.session_state.reset_email, hours=1)
                if attempts >= 5:
                    st.error("Too many requests. Please try again later.")
                else:
                    new_code = generate_reset_code()
                    user_name = get_user_name(st.session_state.reset_email)
                    
                    if store_reset_token(st.session_state.reset_email, new_code) and send_reset_email(st.session_state.reset_email, new_code, user_name):
                        st.success("✅ New code sent!")
                    else:
                        st.error("❌ Failed to resend code.")
        
        st.markdown("#### 🔒 Set New Password")
        new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
        
        # Password strength indicator
        if new_password:
            is_valid, message = validate_password(new_password)
            if is_valid:
                st.success("✅ " + message)
            else:
                st.warning("⚠️ " + message)
        
        # Show password requirements
        with st.expander("📋 Password Requirements"):
            st.markdown("""
            - ✅ At least 8 characters long
            - ✅ Contains uppercase and lowercase letters  
            - ✅ Contains at least one number
            - ✅ Contains at least one special character (!@#$%^&*(),.?":{}|<>)
            """)
        
        if st.button("🔐 Reset Password", key="reset_password", use_container_width=True):
            if not reset_code:
                st.error("Please enter the reset code.")
            elif not new_password:
                st.error("Please enter a new password.")
            elif new_password != confirm_password:
                st.error("Passwords don't match.")
            elif not verify_reset_token(st.session_state.reset_email, reset_code):
                st.error("Invalid or expired reset code. Please request a new one.")
            else:
                # Reset password
                with st.spinner("Resetting password..."):
                    if update_user_password(st.session_state.reset_email, new_password):
                        mark_token_as_used(st.session_state.reset_email, reset_code)
                        st.success("🎉 Password reset successful! You can now login with your new password.")
                        st.balloons()
                        
                        # Reset session state
                        st.session_state.reset_step = 'request'
                        st.session_state.reset_email = ''
                        
                        # Auto redirect to login
                        st.info("Redirecting to login page...")
                        if st.button("🚀 Go to Login", key="go_to_login", use_container_width=True):
                            return 'login'
                    else:
                        st.error("❌ Failed to reset password. Please try again.")
        
        # Back to email entry
        if st.button("← Use Different Email", key="different_email"):
            st.session_state.reset_step = 'request'
            st.session_state.reset_email = ''
            st.rerun()
    
    return None

def display_email_setup_guide():
    """Display email setup guide for administrators"""
    st.markdown("## 📧 Email Configuration Setup Guide")
    
    st.info("To enable password reset functionality, you need to configure email settings.")
    
    # Check current configuration status
    config_status = check_email_config()
    if config_status:
        st.success("✅ Email configuration is properly set up!")
    else:
        st.error("❌ Email configuration is missing or incomplete.")
    
    with st.expander("📋 Gmail Setup Instructions", expanded=not config_status):
        st.markdown("""
        ### 1. Enable 2-Factor Authentication
        - Go to your Google Account settings
        - Navigate to Security → 2-Step Verification
        - Enable 2-Factor Authentication
        
        ### 2. Generate App Password
        - In Security settings, go to "App passwords"
        - Select "Mail" as the app type
        - Generate a 16-character password
        - Copy this password (you won't see it again)
        
        ### 3. Update Environment Variables
        Create a `.env` file in your project root with:
        ```
        SMTP_SERVER=smtp.gmail.com
        SMTP_PORT=587
        SENDER_EMAIL=your-app-email@gmail.com
        SENDER_PASSWORD=your-16-digit-app-password
        APP_NAME=EduPredict
        ```
        """)
    
    with st.expander("🔧 Other Email Providers"):
        st.markdown("""
        ### Outlook/Hotmail
        ```
        SMTP_SERVER=smtp-mail.outlook.com
        SMTP_PORT=587
        SENDER_EMAIL=your-email@outlook.com
        SENDER_PASSWORD=your-password
        ```
        
        ### Yahoo Mail
        ```
        SMTP_SERVER=smtp.mail.yahoo.com
        SMTP_PORT=587
        SENDER_EMAIL=your-email@yahoo.com
        SENDER_PASSWORD=your-app-password
        ```
        
        ### Custom SMTP Server
        ```
        SMTP_SERVER=your-smtp-server.com
        SMTP_PORT=587
        SENDER_EMAIL=your-email@domain.com
        SENDER_PASSWORD=your-password
        ```
        """)
    
    with st.expander("⚠️ Security Best Practices"):
        st.markdown("""
        - **Never commit** your `.env` file to version control
        - Add `.env` to your `.gitignore` file
        - Use environment variables in production
        - Rotate your app passwords regularly
        - Monitor email sending logs for suspicious activity
        - Consider using a dedicated email service for production (SendGrid, Mailgun, etc.)
        """)

# Initialize database tables (run this once)
def initialize_reset_tables():
    """Initialize reset tokens table - call this once during setup"""
    create_reset_tokens_table()
    st.success("Reset tokens table created successfully!")

if __name__ == "__main__":
    # For testing the interface
    forgot_password_interface()