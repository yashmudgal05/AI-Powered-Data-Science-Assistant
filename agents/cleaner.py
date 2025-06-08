import pandas as pd
import numpy as np
from utils.helpers import query_groq

def generate_explanation_with_groq(report: str, data_info: str):
    prompt = f"""
    I am an AI data scientist. The dataset has these characteristics:
    {data_info}

    I performed these cleaning steps:
    {report}

    Please explain each cleaning step in beginner-friendly natural language.
    """

    return query_groq(prompt)

def detect_outliers(df: pd.DataFrame):
    outlier_report = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            outlier_report.append(f"Column '{col}' has {len(outliers)} outliers.")

    return "\n".join(outlier_report)


def clean_data(df: pd.DataFrame):
    cleaning_report = []

    # Step 1: Drop duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    if before != df.shape[0]:
        cleaning_report.append(f"Removed {before - df.shape[0]} duplicate rows.")

    # Step 2: Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    cleaning_report.append("Normalized column names.")

    # Step 3: Fill missing values
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
            cleaning_report.append(f"Filled missing values in '{col}' with median.")
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
            cleaning_report.append(f"Filled missing values in '{col}' with mode.")

    # Step 4: Drop columns with >50% missing
    drop_cols = df.columns[df.isnull().mean() > 0.5]
    if len(drop_cols):
        df.drop(columns=drop_cols, inplace=True)
        cleaning_report.append(f"Dropped columns with >50% missing: {list(drop_cols)}")

    if not cleaning_report:
        cleaning_report.append("No major issues found.")

    # Groq explanation
    data_info = f"{df.shape[0]} rows, {df.shape[1]} columns. Columns: {list(df.columns)}"
    explanation = generate_explanation_with_groq("\n".join(cleaning_report), data_info)

    return df, "\n".join(cleaning_report), explanation


