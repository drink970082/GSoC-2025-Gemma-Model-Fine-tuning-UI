import streamlit as st
import pandas as pd

st.title("Data Explorer")

st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.subheader("Data Preview")
    st.write(data.head())
    
    # Display basic statistics
    st.subheader("Data Statistics")
    st.write(data.describe())
    
    # Display data types
    st.subheader("Data Types")
    st.write(data.dtypes)
    
    # Option to select a column for analysis
    column = st.selectbox("Select a column to analyze", data.columns)
    
    if column:
        st.subheader(f"Unique values in {column}")
        st.write(data[column].unique())