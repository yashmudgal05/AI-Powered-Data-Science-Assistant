# agents/chat_with_data.py

import pandas as pd
from utils.helpers import query_groq
import traceback


def execute_code(df: pd.DataFrame, code: str):
    try:
        result = eval(code, {"df": df, "pd": pd})
        return result, None
    except Exception as e:
        return None, str(e) + "\n" + traceback.format_exc()

def explain_result(question: str, code: str, result):
    prompt = f"""
    I asked: "{question}"

    The code generated was:
    {code}

    Here’s a summary of the result:
    {str(result)[:1000]}

    Please explain in simple terms what this output means.
    """
    return query_groq(prompt)

def generate_code_from_question(question: str, df_sample: str):
    prompt = f"""
    You are a data expert. I will give you:
    - A user's question
    - A preview of a DataFrame (like df.head())
    
    Your job is to return ONLY a short pandas code snippet (no explanation, no print). Assume the DataFrame is named `df`.

    Question: {question}

    DataFrame:
    {df_sample}

    Example format:
    df.groupby('department')['salary'].mean()
    """

    return query_groq(prompt).strip()
