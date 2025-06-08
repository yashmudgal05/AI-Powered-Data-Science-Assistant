# app.py
import streamlit as st
import pandas as pd

# Import all agents
from agents.cleaner import clean_data, detect_outliers
from agents.eda import run_eda, perform_eda
from agents.model_trainer import train_model, auto_train_model
from agents.visuals import plot_barplot, plot_boxplot, plot_correlation_heatmap, plot_numeric_distribution
from agents.chat_with_data import generate_code_from_question, execute_code, explain_result

st.set_page_config(page_title="🧠 AI Data Science Agent", layout="wide")
st.title("🧠 AI Data Science Agent")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Original Data")
    st.write(df.head())

# Step 2: Data Cleaning
if "df_cleaned" not in st.session_state and st.button("🔧 Clean Data"):
    df_cleaned, cleaning_report, explanation = clean_data(df)
    outlier_summary = detect_outliers(df_cleaned)

    # Save intermediate results to session state
    st.session_state["df_cleaned_temp"] = df_cleaned
    st.session_state["cleaning_report"] = cleaning_report
    st.session_state["cleaning_explanation"] = explanation
    st.session_state["outlier_summary"] = outlier_summary

# Show cleaned data and allow approval
if "df_cleaned_temp" in st.session_state:
    st.subheader("🧼 Cleaning Summary")
    st.text_area("Report", st.session_state["cleaning_report"], height=150)
    st.text_area("🧠 Explanation from Groq", st.session_state["cleaning_explanation"], height=250)

    if st.session_state["outlier_summary"]:
        st.warning("⚠️ Outliers Detected")
        st.text_area("Outlier Report", st.session_state["outlier_summary"], height=150)

    st.subheader("📋 Preview Cleaned Data (Unconfirmed)")
    st.dataframe(st.session_state["df_cleaned_temp"].head())

    if st.button("✅ Approve Cleaning and Proceed to EDA"):
        st.session_state["df_cleaned"] = st.session_state["df_cleaned_temp"]
        del st.session_state["df_cleaned_temp"]  # Optional cleanup
        st.success("Approved! Moving to EDA...")

# Step 3: Run EDA on cleaned data
if "df_cleaned" in st.session_state:
    df_cleaned = st.session_state["df_cleaned"]

    if st.button("📊 Run EDA Summary"):
        run_eda(df_cleaned)

    st.subheader("📊 Exploratory Data Analysis (EDA)")
    eda_report, eda_explanation = perform_eda(df_cleaned)
    st.markdown("### 📋 EDA Report")
    st.text_area("Summary", eda_report, height=300)
    st.markdown("### 🧠 EDA Explanation from Groq")
    st.text_area("Explanation", eda_explanation, height=300)

    # Step 4: Visualizations
    st.subheader("📈 Visualize Your Data")
    viz_type = st.selectbox("Choose Visualization Type", [
        "Histogram (Numeric)", 
        "Boxplot (Numeric)", 
        "Barplot (Categorical)", 
        "Correlation Heatmap"
    ])

    if viz_type == "Correlation Heatmap":
        fig = plot_correlation_heatmap(df_cleaned)
        st.pyplot(fig)
    else:
        # Column selector
        if "Numeric" in viz_type:
            columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
        elif "Categorical" in viz_type:
            columns = df_cleaned.select_dtypes(include='object').columns
        else:
            columns = df_cleaned.columns

        selected_col = st.selectbox("Choose Column", columns)

        if viz_type == "Histogram (Numeric)":
            fig = plot_numeric_distribution(df_cleaned, selected_col)
        elif viz_type == "Boxplot (Numeric)":
            fig = plot_boxplot(df_cleaned, selected_col)
        elif viz_type == "Barplot (Categorical)":
            fig = plot_barplot(df_cleaned, selected_col)

        st.pyplot(fig)

    # Step 5: Model Training
    st.subheader("🧠 Auto Model Selector & Trainer")
    target = st.selectbox("🎯 Select Target Column", df_cleaned.columns)

    if st.button("🚀 Train Models"):
        model_name, model_results, model_explanation = auto_train_model(df_cleaned, target)

        st.markdown("### 🧪 Model Evaluation Results")
        for res in model_results:
            st.write(res)

        st.success(f"✅ Best Model: {model_name}")
        st.markdown("### 🧠 Explanation from Groq")
        st.text_area("Model Insight", model_explanation, height=250)

    # Step 6: Chat with Dataset
    st.subheader("💬 Chat with Your Dataset")
    question = st.text_input("Ask a question about your data...")

    if st.button("🧠 Ask"):
        df_preview = df_cleaned.head().to_string()

        with st.spinner("💬 Thinking..."):
            code = generate_code_from_question(question, df_preview)
            st.code(code, language='python')

            result, error = execute_code(df_cleaned, code)
            if error:
                st.error(f"❌ Error:\n{error}")
            else:
                st.markdown("### 📋 Result")
                st.write(result)

                explanation = explain_result(question, code, result)
                st.markdown("### 🧠 Explanation from Groq")
                st.text_area("Insight", explanation, height=250)
