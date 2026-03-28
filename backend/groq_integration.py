import os
from groq import Groq

# AI Model Selection (llama3-8b-8192 is the most stable for Hackathons)
MODEL = "llama3-8b-8192"
CHAT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)
