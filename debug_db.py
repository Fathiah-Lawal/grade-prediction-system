import streamlit as st
import mysql.connector
from mysql.connector import Error

def debug_database_connection():
    """Debug function to test database connection and show detailed error info"""
    
    st.title("🔍 Database Connection Debug")
    
    # Check if secrets exist
    st.subheader("1. Checking Secrets Configuration")
    try:
        host = st.secrets["database"]["host"]
        user = st.secrets["database"]["user"] 
        password = st.secrets["database"]["password"]
        database = st.secrets["database"]["database"]
        port = st.secrets["database"]["port"]
        
        st.success("✅ All secrets found!")
        st.info(f"Host: {host}")
        st.info(f"User: {user}")
        st.info(f"Database: {database}")
        st.info(f"Port: {port}")
        st.info(f"Password: {'*' * len(password)} (hidden)")
        
    except KeyError as e:
        st.error(f"❌ Missing secret: {e}")
        st.info("Make sure you have all these secrets configured in Streamlit Cloud:")
        st.code("""
[database]
host = "switchback.proxy.rlwy.net"
user = "root"
password = "your-railway-password"
database = "railway"
port = 25761
        """)
        return
    
    # Test connection with detailed error handling
    st.subheader("2. Testing Database Connection")
    
    if st.button("🔗 Test Connection"):
        try:
            # Test connection with timeout
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=int(port),
                connection_timeout=10,
                autocommit=True
            )
            
            st.success("✅ Database connection successful!")
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            st.info(f"MySQL Version: {version[0]}")
            
            # Test database access
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            if tables:
                st.info(f"Existing tables: {[table[0] for table in tables]}")
            else:
                st.info("No tables found (this is normal for new database)")
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as e:
            st.error(f"❌ MySQL Error: {e}")
            st.error(f"Error Code: {e.errno}")
            st.error(f"SQL State: {e.sqlstate}")
            
            # Common error solutions
            if e.errno == 1045:
                st.warning("🔧 Solution: Check your username/password")
            elif e.errno == 2003:
                st.warning("🔧 Solution: Check your host/port")
            elif e.errno == 1049:
                st.warning("🔧 Solution: Database doesn't exist, try 'railway' or create it")
            elif e.errno == 2013:
                st.warning("🔧 Solution: Connection timeout - check network/firewall")
                
        except Exception as e:
            st.error(f"❌ General Error: {e}")
            st.error(f"Error Type: {type(e).__name__}")

# Run the debug function
debug_database_connection()