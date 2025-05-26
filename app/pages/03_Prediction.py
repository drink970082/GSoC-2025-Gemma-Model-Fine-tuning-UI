import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate sample data for demonstration
X, y = make_regression(n_samples=100, n_features=1, noise=10)
data = pd.DataFrame(X, columns=['Feature'])
data['Target'] = y

# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(data[['Feature']], data['Target'], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.title("Prediction Page")
st.write("This page allows you to make predictions using a simple linear regression model.")

# User input for prediction
user_input = st.number_input("Enter a value for the feature:", min_value=float(X.min()), max_value=float(X.max()))

# Make prediction
if st.button("Predict"):
    prediction = model.predict([[user_input]])
    st.write(f"The predicted value is: {prediction[0]}")