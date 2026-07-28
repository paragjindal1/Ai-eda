import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(
    page_title="AI Powered Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Powered Data Analyst Agent")
st.write("Upload your dataset or use the default URL to automatically perform EDA, generate visualizations (univariate, bivariate, multivariate), and chat with your data!")

with st.sidebar:
    st.header("Configuration & Input")
    GOOGLE_API_KEY = st.text_input("Google API Key", type="password")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    
    default_url = "https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv"
    data_source = st.selectbox("Choose Data Source", ["Upload CSV/XLSX", "Default URL"])
    
    file_path = default_url
    uploaded_file = None
    
    if data_source == "Upload CSV/XLSX":
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                file_path = uploaded_file
            else:
                file_path = uploaded_file

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please provide both Google API Key and Groq API Key in the sidebar to run the agent.")
else:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )

    groq_llm = ChatGroq(
        model="qwen-2.5-coder-32b-instruct",
        api_key=GROQ_API_KEY
    )

    def temp_tool():
        """This is just a dummy tool"""
        return "Hello world"

    agent = create_agent(
        model=gemini_llm,
        tools=[temp_tool]
    )

    def load_dataset(path):
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(path)

    try:
        df = load_dataset(file_path)
        st.success("Dataset Loaded Successfully!")
        
        with st.expander("🔍 Preview Raw Dataset"):
            st.dataframe(df.head(10))
            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = None

    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Auto EDA & Analysis", "📊 Charts & Visualizations", "💬 Chat with Data"])

        with tab1:
            st.subheader("Basic & Advanced Exploratory Data Analysis")
            
            if st.button("Run AI EDA Analysis"):
                with st.spinner("AI is analyzing your dataset structure..."):
                    try:
                        sample_df = df.sample(min(5, len(df)))
                        prompt = f"""You are a data analyst. Write a python function named `perform_eda(df)` that returns basic EDA info like shape, missing values, and columns. 
                        Data frame sample: {sample_df}
                        Data stats: {sample_df.describe()}
                        Return ONLY executable python code inside ```python and ``` blocks."""
                        
                        response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                        ans = response["messages"][-1].content
                        if isinstance(ans, list):
                            ans = ans[0].get('text', str(ans))
                        
                        if "```python" in ans:
                            code = ans.split("```python")[1].split("```")[0]
                        elif "```" in ans:
                            code = ans.split("```")[1].split("```")[0]
                        else:
                            code = ans
                            
                        with open('basic_eda.py', 'w') as f:
                            f.write(code)
                    except Exception as e:
                        st.error(f"Error generating basic EDA: {e}")

                with st.spinner("AI is generating comprehensive Advanced EDA..."):
                    try:
                        advance_prompt = """Give detailed python code for advanced data analysis including describe, correlation, univariate numerical and object column analysis, bivariate analysis, time series (if date column exists), and multivariate analysis using plots like bar plot with hue. Write it inside a single function eda_by_ai(df). Return ONLY python code inside ```python and ``` blocks."""
                        
                        response = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
                        ans = response["messages"][-1].content
                        if isinstance(ans, list):
                            ans = ans[0].get('text', str(ans))
                            
                        if "```python" in ans:
                            code = ans.split("```python")[1].split("```")[0]
                        elif "```" in ans:
                            code = ans.split("```")[1].split("```")[0]
                        else:
                            code = ans
                            
                        with open('advance_eda.py', 'w') as f:
                            f.write(code)
                            
                        st.success("EDA scripts successfully created and executed!")
                    except Exception as e:
                        st.error(f"Error generating advanced EDA: {e}")

            if os.path.exists('basic_eda.py'):
                st.write("### Basic EDA Results")
                try:
                    from basic_eda import perform_eda
                    st.text(perform_eda(df) if callable(perform_eda) else "Executed successfully")
                except Exception as e:
                    st.info(f"Basic EDA module loaded. Details: {e}")
                    st.write(df.describe())

            if os.path.exists('advance_eda.py'):
                st.write("### Advanced EDA Visualizations & Insights")
                try:
                    from advance_eda import eda_by_ai
                    fig, ax = plt.subplots(figsize=(10, 5))
                    eda_by_ai(df)
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Displaying correlation heatmap as part of fallback advanced analysis:")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)
                    else:
                        st.write("No numeric columns available for correlation.")

        with tab2:
            st.subheader("Auto-Generated Univariate, Bivariate & Multivariate Charts")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Univariate Analysis")
                if numeric_cols:
                    selected_num = st.selectbox("Select Numeric Column for Distribution", numeric_cols)
                    fig, ax = plt.subplots()
                    sns.histplot(df[selected_num], kde=True, ax=ax, color='skyblue')
                    st.pyplot(fig)
                
                if categorical_cols:
                    selected_cat = st.selectbox("Select Categorical Column for Count", categorical_cols)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.countplot(data=df, x=selected_cat, order=df[selected_cat].value_counts().index[:10], ax=ax, palette="viridis")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with col2:
                st.markdown("#### Bivariate & Multivariate Analysis")
                if len(numeric_cols) >= 2:
                    num_x = st.selectbox("X-axis Numeric", numeric_cols, index=0)
                    num_y = st.selectbox("Y-axis Numeric", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                    hue_col = st.selectbox("Hue / Group By (Optional)", [None] + categorical_cols)
                    
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=num_x, y=num_y, hue=hue_col, ax=ax, palette="Set2")
                    st.pyplot(fig)

        with tab3:
            st.subheader("💬 Chat with Your Dataset")
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_query = st.text_input("Ask anything about your data (e.g., 'What is the total sales?', 'Show rows where profit is negative'):")
            
            if st.button("Ask Agent"):
                if user_query:
                    with st.spinner("AI Analyst is processing your query..."):
                        chat_prompt = f"""You are a helpful data analyst agent. Given the dataframe `df` with columns {list(df.columns)}, 
                        answer the following user query by writing python code or giving direct insights.
                        User Query: {user_query}
                        DataFrame Info: {df.info()}
                        """
                        response = agent.agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]}) if hasattr(agent, 'agent') else agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                        ans = response["messages"][-1].content
                        if isinstance(ans, list):
                            ans = ans[0].get('text', str(ans))
                        
                        st.session_state.chat_history.append({"user": user_query, "bot": ans})
                else:
                    st.warning("Please enter a query.")

            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"**👤 User:** {chat['user']}")
                st.markdown(f"**🤖 Agent:** {chat['bot']}")
                st.markdown("---")
