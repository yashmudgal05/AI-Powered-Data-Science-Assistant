import os
import requests
from dotenv import load_dotenv
# Load environment variables from the .env file in the parent directory
load_dotenv()

def get_groq_api_key():
    return os.getenv("GROQ_API_KEY")  # Or hardcode if testing


def query_groq(prompt, model="llama3-8b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {get_groq_api_key()}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert data scientist."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq explanation failed: {e}"
