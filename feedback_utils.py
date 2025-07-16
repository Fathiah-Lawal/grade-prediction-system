# feedback_utils.py
import re
import streamlit as st
import os
import requests
import json

# Load Gemini API key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Use Gemini 1.5 Flash for better quota and speed
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def generate_feedback(predicted_class: str, input_dict: dict) -> str:
    if not GEMINI_API_KEY:
        return "❌ No Gemini API key found. Please add it to secrets.toml."

    # Construct the detailed prompt
    prompt = (
        f"You are a supportive academic advisor who gives detailed and practical guidance.\n"
        f"The student’s predicted grade is: {predicted_class}.\n"
        f"Here are their current attributes:\n"
        f"- Study Hours/Week: {input_dict.get('Hours_Studied', 'N/A')}\n"
        f"- Previous Scores: {input_dict.get('Previous_Scores', 'N/A')}\n"
        f"- Attendance: {input_dict.get('Attendance', 'N/A')}%\n"
        f"- Sleep Hours/Night: {input_dict.get('Sleep_Hours', 'N/A')}\n"
        f"- Tutoring Sessions/Week: {input_dict.get('Tutoring_Sessions', 'N/A')}\n"
        f"- Internet Access: {input_dict.get('Internet_Access', 'N/A')}\n"
        f"- Learning Disabilities: {input_dict.get('Learning_Disabilities', 'N/A')}\n"
        f"- Motivation Level: {input_dict.get('Motivation_Level', 'N/A')}\n"
        f"- Peer Influence: {input_dict.get('Peer_Influence', 'N/A')}\n"
        f"- Access to Resources: {input_dict.get('Access_to_Resources', 'N/A')}\n\n"
        "Please write 3 very detailed and personalized suggestions for this student. "
        "Each suggestion should be around 4–6 lines long, practical, and clearly structured. "
        "Use an encouraging tone, and end with a motivating closing message to the student."
    )

    headers = {"Content-Type": "application/json"}
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

        # Handle common error formats
        if "candidates" not in result:
            return f"⚠️ Gemini error: {result.get('error', {}).get('message', 'Unknown issue')}"

        feedback = result["candidates"][0]["content"]["parts"][0]["text"]
        feedback = feedback.strip()
        feedback = re.sub(r'\n{3,}', '\n\n', feedback)  # clean triple newlines
        return feedback

    except Exception as e:
        return f"❌ Error generating feedback: {e}"
