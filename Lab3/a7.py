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