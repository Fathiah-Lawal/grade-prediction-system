# feedback_utils.py
import re
from ollama import Client

# Initialize Ollama client 
ollama_client = Client()

def generate_feedback(predicted_class: str, input_dict: dict) -> str:
    """
    Generate actionable feedback for a student based on predicted grade and input features.

    Parameters:
    - predicted_class (str): The predicted grade/class of the student.
    - input_dict (dict): Dictionary containing student features.

    Returns:
    - feedback (str): A string containing 3 specific actionable tips.
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
        "Please provide 3 concise, actionable tips to help this student improve their performance. Talk to the student like you're interacting with him/her personally"
    )

    try:
        response = ollama_client.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        feedback = response['message']['content']
    except Exception as e:
        return f"Error generating feedback: {e}"

    # Debug print of raw response; remove or comment out in production
    print("RAW FEEDBACK:", repr(feedback))

   # Fix vertical text if it's likely character-separated
    lines = feedback.splitlines()
    if sum(len(line.strip()) <= 2 for line in lines) > len(lines) * 0.5:
        feedback = feedback.replace('\n', '')


    # Then collapse multiple blank lines to a single blank line
    feedback = re.sub(r'\n{2,}', '\n\n', feedback).strip()

    return feedback
