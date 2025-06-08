# agents/model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from utils.helpers import query_groq
import streamlit as st

def auto_train_model(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    if y.dtype == "object" or len(y.unique()) <= 10:
        y = LabelEncoder().fit_transform(y)
        task_type = "classification"
    else:
        task_type = "regression"

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForestClassifier": RandomForestClassifier()
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor()
        }

    best_model = None
    best_score = -float("inf")
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)

        results.append(f"{name}: Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name

    # Get explanation from Groq
    explanation = query_groq(f"""
    I trained multiple models for a {task_type} task. The target column was '{target_col}'.
    Here are the results:
    {'; '.join(results)}.
    The best model was {best_model_name} with score {best_score:.4f}.
    Explain in simple language why this model might be a good choice and what it could mean.
    """)

    return best_model_name, results, explanation

def train_model(df: pd.DataFrame):
    st.subheader("🧠 Auto Model Selector & Trainer")

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Train Models"):
        model_name, model_results, model_explanation = auto_train_model(df, target)

        st.markdown("### 🧪 Model Evaluation Results")
        for res in model_results:
            st.write(res)

        st.success(f"✅ Best Model: {model_name}")
        st.markdown("### 🧠 Explanation from Groq")
        st.text_area("Model Insight", model_explanation, height=250)