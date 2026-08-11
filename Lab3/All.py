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


import matplotlib.pyplot as plt
from scipy.spatial import distance


def calculate_minkowski(vec1, vec2, p):
    if len(vec1) != len(vec2):
        raise ValueError()
    return sum(abs(a - b) ** p for a, b in zip(vec1, vec2)) ** (1 / p)

features = ['Income', 'Recency', 'MntWines']
data_matrix = df[features].values.tolist()

vec_A = data_matrix[0]
vec_B = data_matrix[1]

p_values = list(range(1, 11))
distances = [calculate_minkowski(vec_A, vec_B, p) for p in p_values]

scipy_dist = distance.minkowski(vec_A, vec_B, p=3)
print(calculate_minkowski(vec_A, vec_B, 3))
print(scipy_dist)

plt.plot(p_values, distances, marker='o')
plt.title("Minkowski Distance vs. p-value")
plt.xlabel("p (Order parameter)")
plt.ylabel("Distance")
plt.show()


import numpy as np

def calculate_dot_product(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))

def calculate_euclidean_norm(vec):
    return sum(x ** 2 for x in vec) ** 0.5

v_a = [1, 2, 3]
v_b = [4, 5, 6]

custom_dot = calculate_dot_product(v_a, v_b)
numpy_dot = np.dot(v_a, v_b)
print(f"Dot Product -> Custom: {custom_dot}, NumPy: {numpy_dot}")

custom_norm = calculate_euclidean_norm(v_a)
numpy_norm = np.linalg.norm(v_a)
print(f"Norm -> Custom: {custom_norm}, NumPy: {numpy_norm}")