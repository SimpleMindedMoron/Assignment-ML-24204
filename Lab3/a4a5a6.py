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