# 🧠 AI-Powered Data Science Assistant

A one-stop AI-powered platform that automates the entire data science workflow — from data cleaning to EDA to model training and chatting with your dataset — using Streamlit, pandas, scikit-learn, and Groq LLMs.

🚀 Turn raw data into insights and models in minutes.  
Ideal for beginners, analysts, and ML professionals alike.

---

## 📦 Features

✅ **Dataset Upload**  
Upload CSV files (support for large files up to millions of rows).

🧼 **Automated Data Cleaning**  
- Handles missing values
- Detects outliers
- Removes duplicates
- Shows detailed cleaning reports

📊 **Exploratory Data Analysis (EDA)**  
- Statistical summaries (mean, std, mode, etc.)
- Auto-generated insights using Groq LLM
- Dataset shape, column types, and distribution info

📈 **Visualization Suite**  
- Histogram & Boxplot (numeric)
- Barplot (categorical)
- Correlation heatmap

🤖 **Auto Model Trainer**  
- Detects classification vs regression
- Trains Logistic/Linear/RandomForest models
- Compares performance metrics
- Groq-generated model selection insights

💬 **Chat with Your Data**  
- Ask questions in plain English
- Get Python code + output + AI explanation

---

## 🔧 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/data-science-assistant.git
cd data-science-assistant
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Set Up API Key for Groq
Create a .env file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
4. Run the App
```bash
streamlit run app.py
```

🚧 Coming Soon
🔜 Hyperparameter Tuning Agent – Smart optimization via Grid/Random/LLM search
🔜 Custom Model Uploads – Bring your own model (ML/DL class)
🔜 Model Interpretability – SHAP / LIME for explainability
🔜 Multi-Turn Chat Memory – Intelligent memory during dataset Q&A
🔜 Exportable Reports – One-click download of cleaning/EDA/model results
🔜 Model Download – Save and export best trained model in .pkl format
