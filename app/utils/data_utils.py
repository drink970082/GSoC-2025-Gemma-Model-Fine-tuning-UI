def load_data(file_path):
    import pandas as pd
    
    # Load the dataset from the specified file path
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    # Perform basic data cleaning
    data = data.dropna()  # Remove missing values
    return data

def preprocess_data(file_path):
    # Load and clean the data
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    return cleaned_data