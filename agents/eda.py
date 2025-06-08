# agents/eda.py

import pandas as pd
from utils.helpers import query_groq
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def generate_groq_eda_explanation(numeric_summary, categorical_summary, shape):
    prompt = f"""
    I have a dataset with shape: {shape[0]} rows and {shape[1]} columns.

    Here is the numeric column summary (mean, std, min, max etc):
    {numeric_summary}

    And here is the categorical column summary (mode and unique counts):
    {categorical_summary}

    Please give a human-readable explanation of interesting patterns, distributions, or warnings.
    """

    return query_groq(prompt)


def perform_eda(df: pd.DataFrame):
    eda_report = []
    numeric_summary = ""
    categorical_summary = ""

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric Columns Summary
    if numeric_cols:
        numeric_summary_df = df[numeric_cols].describe().T
        numeric_summary = numeric_summary_df.to_string()
        eda_report.append("📊 **Numeric Column Summary:**\n" + numeric_summary)

    # Categorical Columns Summary
    if categorical_cols:
        cat_summary = []
        for col in categorical_cols:
            mode = df[col].mode()[0]
            unique_vals = df[col].nunique()
            cat_summary.append(f"{col}: Mode = {mode}, Unique = {unique_vals}")
        categorical_summary = "\n".join(cat_summary)
        eda_report.append("🔤 **Categorical Column Summary:**\n" + categorical_summary)

    # Ask Groq for explanation
    explanation = generate_groq_eda_explanation(numeric_summary, categorical_summary, df.shape)

    return "\n\n".join(eda_report), explanation

def run_eda(df: pd.DataFrame):
    st.subheader("📉 Quick EDA Visuals")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns to display.")
        return

    st.write("Here’s a quick distribution of all numeric features:")

    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)
