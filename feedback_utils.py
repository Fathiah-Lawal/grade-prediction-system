# feedback_utils.py
import re
import streamlit as st
import os
import requests
import json

# Get Gemini API key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Gemini endpoint for chat-style prompt
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

def generate_feedback(predicted_class: str, input_dict: dict) -> str:
    """
    Generate actionable feedback for a student based on predicted grade and input features.
    """

    prompt = (
        f"You are an academic advisor.\n"
        f"Predicted Grade: {predicted_class}\n"
        f"Study Hours/Week: {input_dict.get('Hours_Studied', 'N/A')}\n"
        f"Previous Scores: {input_dict.get('Previous_Scores', 'N/A')}\n"
        f"Attendance: {input_dict.get('Attendance', 'N/A')}%\n"
        f"Sleep Hours/Night: {input_dict.get('Sleep_Hours', 'N/A')}\n"
        f"Tutoring Sessions/Week: {input_dict.get('Tutoring_Sessions', 'N/A')}\n"
        f"Internet Access: {input_dict.get('Internet_Access', 'N/A')}\n"
        f"Learning Disabilities: {input_dict.get('Learning_Disabilities', 'N/A')}\n"
        f"Motivation Level: {input_dict.get('Motivation_Level', 'N/A')}\n"
        f"Peer Influence: {input_dict.get('Peer_Influence', 'N/A')}\n"
        f"Access to Resources: {input_dict.get('Access_to_Resources', 'N/A')}\n\n"
        "Please provide 3 concise, actionable tips to help this student improve their performance. Talk to the student like you're interacting with him/her personally."
    )

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data)
        result = response.json()
        feedback = result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Error generating feedback: {e}"

    # Clean up feedback
    feedback = feedback.strip()
    feedback = re.sub(r'\n{2,}', '\n\n', feedback)

    return feedback
