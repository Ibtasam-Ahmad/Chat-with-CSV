import pandas as pd
import streamlit as st
from pandasai.llm import OpenAI
from pandasai import SmartDataframe
import warnings

# Ignore warnings
warnings.simplefilter(action='ignore', category=Warning)

# Set up OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Initialize the LLM when the key is provided
llm = None
if api_key:
    llm = OpenAI(api_token=api_key)

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None and llm:
    # Read the uploaded CSV into a dataframe
    df_order = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataframe
    st.subheader("Data Preview")
    st.write(df_order.head())
    
    # Create a SmartDataframe
    sdf = SmartDataframe(df_order, config={"llm": llm})

    # User input for query
    query = st.text_input("Ask a query about the data (e.g., 'What is the total profit?')")

    if query:
        # Run the query on the SmartDataframe
        response = sdf.chat(query)
        
        # Display the response
        st.subheader("Query Result:")
        st.write(response)

        # If the query is a plot, it might return a chart URL, so we handle that.
        if isinstance(response, str) and response.endswith('.png'):
            st.image(response)
        elif isinstance(response, pd.DataFrame):
            st.dataframe(response)

else:
    st.warning("Please upload a CSV file and provide an OpenAI API key to interact with the data.")
