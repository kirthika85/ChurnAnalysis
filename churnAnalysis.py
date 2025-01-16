import streamlit as st
import openai
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gorilla import GorillaClient
import re

# Set up NLTK data path
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Streamlit app title
st.title("SaaS Churn Analysis and Prediction")

# Step 1: User Inputs
st.sidebar.header("User Inputs")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
gorilla_api_key = st.sidebar.text_input("Gorilla API Key", type="password")
company_name = st.sidebar.text_input("Enter SaaS Company Name", "Salesforce")
competitor_name = st.sidebar.text_input("Enter Competitor Name", "HubSpot")
timeframe = st.sidebar.text_input("Enter Timeframe (e.g., 'last 6 months')", "last 6 months")

# Initialize APIs
if openai_api_key:
    openai.api_key = openai_api_key
if gorilla_api_key:
    gorilla_client = GorillaClient(api_key=gorilla_api_key)

# Function to fetch competitor data using Gorilla API
def fetch_gorilla_data(company, competitor):
    try:
        query = f"churn reasons for {company} to {competitor}"
        response = gorilla_client.query(query)
        return response
    except Exception as e:
        st.error(f"Error fetching Gorilla data: {e}")
        return []

# Function to analyze churn reasons using OpenAI
def analyze_churn(data, company, competitor):
    try:
        prompt = f"""
        Analyze the following data about customer churn from {company} to {competitor}. Identify reasons for churn, potential warning signs, and summarize the key insights:

        Data: {data}

        Output format:
        - Reasons for churn
        - Warning signs (e.g., customer sentiment, stock trends, market indicators)
        - Conclusion
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        st.error(f"Error using OpenAI: {e}")
        return ""

# Function to visualize results
def visualize_results(churn_reasons):
    df = pd.DataFrame(churn_reasons, columns=["Reason", "Count"])
    st.bar_chart(df.set_index("Reason"))

# Fetch and Analyze Data
if st.button("Analyze Churn"):
    if not (openai_api_key and gorilla_api_key):
        st.error("Please provide both OpenAI and Gorilla API keys.")
    else:
        # Fetch data from Gorilla API
        st.write(f"Fetching data for {company_name} and {competitor_name}...")
        data = fetch_gorilla_data(company_name, competitor_name)

        if data:
            st.write("Data fetched successfully! Analyzing reasons for churn...")
            analysis = analyze_churn(data, company_name, competitor_name)

            if analysis:
                st.subheader("Analysis")
                st.write(analysis)

                # Extract churn reasons for visualization
                churn_reasons = re.findall(r"- (.*?): \d+", analysis)
                reason_counts = [(reason, analysis.count(reason)) for reason in set(churn_reasons)]

                st.subheader("Visualization")
                visualize_results(reason_counts)
        else:
            st.error("No data available from Gorilla API.")
