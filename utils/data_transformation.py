import pandas as pd
import json
import random
import os
from pathlib import Path

def extract_data(json_data):
    """
    Extracts input_data and output_data from response field in JSON data.
    
    Args:
        json_data (list): List of JSON objects containing 'prompt' and 'response'
        
    Returns:
        DataFrame: DataFrame with columns 'input_data' and 'output_data'
    """
    extracted_data = []
    for item in json_data:
        try:
            response = json.loads(item['response'])
            input_data = response.get('input_data')
            output_data = response.get('output_data')
            extracted_data.append({'input_data': input_data, 'output_data': output_data})
        except json.JSONDecodeError:
            print("Error decoding JSON in response field")
    
    return pd.DataFrame(extracted_data)

def manual_train_test_split(data, test_size=0.1):
    """
    Splits data into training and testing sets without sklearn.
    
    Args:
        data (DataFrame): Data to split
        test_size (float): Proportion of the data to use as test set
    
    Returns:
        Tuple: train and test DataFrames
    """
    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split index
    split_index = int(len(data) * (1 - test_size))
    
    # Split the data
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    
    return train_data, test_data

def save_data(train_data, test_data, train_dir, test_dir):
    """
    Saves train and test DataFrames to JSONL files.
    
    Args:
        train_data (DataFrame): Training data
        test_data (DataFrame): Testing data
        train_dir (str): Directory to save training data
        test_dir (str): Directory to save testing data
    """
    train_path = os.path.join(train_dir, 'train.jsonl')
    test_path = os.path.join(test_dir, 'test.jsonl')

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    train_data.to_json(train_path, orient='records', lines=True)
    test_data.to_json(test_path, orient='records', lines=True)

    print(f"Training data saved to {train_path}")
    print(f"Test data saved to {test_path}")

def main():

    input_file = 'data/extracted_data/menu_examples.json'
    train_dir = 'data/train_data'
    test_dir = 'data/test_data'

    # Load data
    with open(input_file, 'r') as f:
        json_data = json.load(f)
    
    # Extract input_data and output_data
    data_df = extract_data(json_data)
    
    # Manual train-test split
    train_df, test_df = manual_train_test_split(data_df)
    
    # Save train and test data
    save_data(train_df, test_df, train_dir, test_dir)

# # Example usage
# if __name__ == "__main__":
#     input_file = 'data/extracted_data/menu_examples.json'  # Replace with your input file path
#     train_dir = 'data/train_data'
#     test_dir = 'data/test_data'
    
#     main(input_file, train_dir, test_dir)
