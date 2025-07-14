# admin.py

import streamlit as st
import pandas as pd
import os
import datetime
import plotly.express as px
from db import connect_to_db

def admin_dashboard():
    if not st.session_state.get("logged_in") or st.session_state.get("user_role") != "Educator":
        st.error("Access denied. Admins only.")
        return

    st.header("📊 Admin Dashboard")

    # Get user count from database
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) AS total_users FROM users")
    total_users = cursor.fetchone()["total_users"]
    conn.close()

    # Get prediction count from CSV files
    try:
        result_files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.csv')]
        total_predictions = 0
        active_users = 0
        
        for file in result_files:
            try:
                df = pd.read_csv(file)
                if len(df) > 0:
                    total_predictions += len(df)
                    active_users += 1
            except:
                continue
                
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", total_users)
        with col2:
            st.metric("Total Predictions", total_predictions)
        with col3:
            st.metric("Active Users", active_users)
            
    except Exception as e:
        st.error(f"Error reading prediction files: {e}")
        st.metric("Total Users", total_users)
        st.metric("Total Predictions", "Error")

def manage_users():
    if not st.session_state.get("logged_in") or st.session_state.get("user_role") != "Educator":
        st.error("Access denied. Admins only.")
        return

    st.header("👥 Manage Users")

    # --- Filters ---
    with st.expander("🔍 Search & Filter", expanded=True):
        search_query = st.text_input("Search by email:")
        role_filter = st.selectbox("Filter by role:", ["All", "Student", "Educator"])

    # --- Fetch Users ---
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT id, email, role FROM users"
    cursor.execute(query)
    users = cursor.fetchall()
    conn.close()

    # --- Apply Filters ---
    if search_query:
        users = [u for u in users if search_query.lower() in u["email"].lower()]
    if role_filter != "All":
        users = [u for u in users if u["role"] == role_filter]

    # --- Display Users ---
    if not users:
        st.info("No users found.")
        return

    for user in users:
        col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
        col1.write(user["email"])
        col2.write(user["role"])
        
        # Show prediction count for this user
        user_predictions = 0
        try:
            user_file = f"results_{user['email']}.csv"
            if os.path.exists(user_file):
                df = pd.read_csv(user_file)
                user_predictions = len(df)
        except:
            pass
        col3.write(f"{user_predictions} predictions")
        
        if col4.button("Delete", key=f"del_{user['id']}"):
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = %s", (user["id"],))
            conn.commit()
            conn.close()
            
            # Also delete user's prediction file
            try:
                user_file = f"results_{user['email']}.csv"
                if os.path.exists(user_file):
                    os.remove(user_file)
            except:
                pass
                
            st.success(f"Deleted {user['email']} and their predictions")
            st.rerun()

def all_predictions():
    if not st.session_state.get("logged_in") or st.session_state.get("user_role") != "Educator":
        st.error("Access denied. Admins only.")
        return

    st.header("📈 All User Predictions")
    
    try:
        # Get all result files
        result_files = [f for f in os.listdir('.') if f.startswith('results_') and f.endswith('.csv')]
        
        if not result_files:
            st.warning("No prediction files found. Users need to make predictions first.")
            st.info("💡 **How predictions are stored:**")
            st.write("- When users make predictions, they're saved as CSV files")
            st.write("- Files are named: `results_[user_email].csv`")
            st.write("- Each user has their own prediction file")
            return
        
        # Display file count
        st.success(f"Found {len(result_files)} user prediction files")
        
        all_data = []
        file_info = []
        
        for file in result_files:
            try:
                # Extract email from filename
                user_email = file.replace('results_', '').replace('.csv', '')
                
                # Read the file
                df = pd.read_csv(file)
                
                if len(df) > 0:
                    # Add user email to each record
                    df['User_Email'] = user_email
                    all_data.append(df)
                    
                    # Store file info
                    file_info.append({
                        'User_Email': user_email,
                        'Predictions': len(df),
                        'Latest_Prediction': df['Timestamp'].max() if 'Timestamp' in df.columns else 'Unknown'
                    })
                    
            except Exception as e:
                st.warning(f"Could not read {file}: {str(e)}")
                continue
        
        if not all_data:
            st.error("No valid prediction data found in any files.")
            st.info("**Possible reasons:**")
            st.write("- Users haven't made any predictions yet")
            st.write("- CSV files are corrupted or empty")
            st.write("- File format is incorrect")
            
            # Show debug info
            with st.expander("🔍 Debug Information"):
                st.write("**Files found:**")
                for file in result_files:
                    st.write(f"- {file}")
            return
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(combined_df))
        with col2:
            st.metric("Active Users", len(file_info))
        with col3:
            if 'Confidence' in combined_df.columns:
                avg_confidence = combined_df['Confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            else:
                st.metric("Avg Confidence", "N/A")
        with col4:
            if 'Predicted_Grade' in combined_df.columns:
                most_common_grade = combined_df['Predicted_Grade'].mode().iloc[0]
                st.metric("Most Common Grade", most_common_grade)
            else:
                st.metric("Most Common Grade", "N/A")
        
        # Show user activity summary
        st.markdown("### 👥 User Activity Summary")
        file_info_df = pd.DataFrame(file_info)
        st.dataframe(file_info_df, use_container_width=True)
        
        # Display the main data with filters
        st.markdown("### 📋 All Predictions")
        
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            if 'User_Email' in combined_df.columns:
                selected_users = st.multiselect(
                    "Filter by Users (leave empty for all):",
                    options=sorted(combined_df['User_Email'].unique().tolist()),
                    default=[]
                )
                
                if selected_users:
                    combined_df = combined_df[combined_df['User_Email'].isin(selected_users)]
        
        with col2:
            if 'Predicted_Grade' in combined_df.columns:
                selected_grades = st.multiselect(
                    "Filter by Predicted Grades (leave empty for all):",
                    options=sorted(combined_df['Predicted_Grade'].unique().tolist()),
                    default=[]
                )
                
                if selected_grades:
                    combined_df = combined_df[combined_df['Predicted_Grade'].isin(selected_grades)]
        
        # Show filtered results
        if len(combined_df) > 0:
            st.dataframe(combined_df, use_container_width=True)
            
            # Visualizations
            if 'Predicted_Grade' in combined_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Grade distribution
                    grade_counts = combined_df['Predicted_Grade'].value_counts()
                    fig_pie = px.pie(
                        values=grade_counts.values,
                        names=grade_counts.index,
                        title="Overall Grade Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # User activity
                    if 'User_Email' in combined_df.columns:
                        user_counts = combined_df['User_Email'].value_counts()
                        fig_bar = px.bar(
                            x=user_counts.values,
                            y=user_counts.index,
                            orientation='h',
                            title="Predictions per User",
                            labels={'x': 'Number of Predictions', 'y': 'User Email'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No data matches the selected filters.")
        
        # Download option
        csv_data = combined_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Predictions",
            data=csv_data,
            file_name=f"all_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        
        # Debug information
        with st.expander("🔍 Show Error Details"):
            st.write("**Error details:**")
            st.write(str(e))
            
            st.write("**Directory contents:**")
            try:
                all_files = os.listdir('.')
                csv_files = [f for f in all_files if f.endswith('.csv')]
                st.write("CSV files found:")
                for f in csv_files:
                    st.write(f"- {f}")
            except Exception as debug_e:
                st.write(f"Cannot list directory: {debug_e}")