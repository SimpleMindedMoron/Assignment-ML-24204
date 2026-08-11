import pandas as pd

def label_encode(column_data):
    unique_vals = list(set(column_data))
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    return [mapping[val] for val in column_data]

def one_hot_encode(column_data):
    unique_vals = list(set(column_data))
    encoded_data = []
    for val in column_data:
        row = [1 if val == u else 0 for u in unique_vals]
        encoded_data.append(row)
    return encoded_data

df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
df.dropna(subset=['Income'], inplace=True) 

df['Education_Encoded'] = label_encode(df['Education'].tolist())
marital_one_hot = one_hot_encode(df['Marital_Status'].tolist())