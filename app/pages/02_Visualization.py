import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample data
data = pd.read_csv('data/sample_data.csv')

st.title('Data Visualization')

# Sidebar for selecting visualization type
st.sidebar.header('Select Visualization Type')
plot_type = st.sidebar.selectbox('Choose a plot type:', ['Line Plot', 'Bar Plot', 'Scatter Plot'])

# Function to create visualizations
def create_visualization(plot_type):
    if plot_type == 'Line Plot':
        st.subheader('Line Plot')
        plt.figure(figsize=(10, 5))
        plt.plot(data['feature'], data['value'])  # Replace with actual column names
        plt.title('Line Plot of X vs Y')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        st.pyplot(plt)

    elif plot_type == 'Bar Plot':
        st.subheader('Bar Plot')
        plt.figure(figsize=(10, 5))
        sns.barplot(x='categorvalue', y='value_column', data=data)  # Replace with actual column names
        plt.title('Bar Plot of Categories')
        st.pyplot(plt)

    elif plot_type == 'Scatter Plot':
        st.subheader('Scatter Plot')
        plt.figure(figsize=(10, 5))
        plt.scatter(data['feature'], data['value'])  # Replace with actual column names
        plt.title('Scatter Plot of X vs Y')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        st.pyplot(plt)

# Create the selected visualization
create_visualization(plot_type)