import os
import pandas as pd

directory = 'csv'

def create_csv(file_name, column_1, column_2):
    path = get_full_path(file_name)
    if not os.path.isfile(path):
        # Create a DataFrame with two columns
        df = pd.DataFrame(columns=[column_1, column_2])
        # Save the DataFrame to a CSV file
        df.to_csv(path, index=False)
        print(f'{path} created with 2 columns.')
    else:
        print(f'{path} already exists.')

def insert_row(file_name, column_1_value, column_2_value):
    path = get_full_path(file_name)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        new_row = pd.DataFrame({df.columns[0]: [column_1_value], df.columns[1]: [column_2_value]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(path, index=False)
    else:
        print(f'Error: {path} does not exist. Please create the file first.')

def get_full_path(file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Directory {directory} created.')
    full_path = os.path.join(directory, file_name)
    return full_path