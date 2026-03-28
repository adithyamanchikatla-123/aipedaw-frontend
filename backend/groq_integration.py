import os
from groq import Groq

# AI Model Selection (Latest supported models as of March 2026)
MODEL = "llama-3.1-8b-instant"
CHAT_MODEL = "llama-3.3-70b-versatile"

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)
