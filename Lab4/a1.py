#Q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Generated using Gemini AI
def label_encode(column_data):
    """Encodes categorical data into numeric labels."""
    unique_vals = list(set(column_data))
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    return [mapping[val] for val in column_data]

# Generated using Gemini AI
def calculate_minkowski(vec1, vec2, p):
    """Calculates Minkowski distance for order p."""
    if len(vec1) != len(vec2):
        raise ValueError("Vector dimensions must match.")
    return sum(abs(a - b) ** p for a, b in zip(vec1, vec2)) ** (1 / p)

# Generated using Gemini AI
def kmeans_clustering(data, k, max_iterations=100):
    """Performs K-Means clustering on the provided data."""
    num_features = len(data[0])
    centroids = random.sample(data, k)
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            # Using the Minkowski function with p=2 for Euclidean
            distances = [calculate_minkowski(point, c, 2) for c in centroids]
            closest_idx = distances.index(min(distances))
            clusters[closest_idx].append(point)
            
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                continue
            cluster_mean = [sum(pt[f] for pt in cluster) / len(cluster) for f in range(num_features)]
            new_centroids.append(cluster_mean)
            
        if centroids == new_centroids:
            break
            
        centroids = new_centroids
        
    return clusters, centroids

#Q2
import unittest

# Generated using Gemini AI
class TestLabFunctions(unittest.TestCase):

    def test_label_encode(self):
        data = ['Cat', 'Dog', 'Cat', 'Bird']
        encoded = label_encode(data)
        self.assertEqual(len(encoded), 4)
        self.assertEqual(encoded[0], encoded[2]) # 'Cat' should have the same label
        self.assertNotEqual(encoded[0], encoded[1])

    def test_calculate_minkowski_manhattan(self):
        v1 = [1, 2, 3]
        v2 = [4, 0, 3]
        # Manhattan (p=1): |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
        dist = calculate_minkowski(v1, v2, 1)
        self.assertAlmostEqual(dist, 5.0)

    def test_calculate_minkowski_euclidean(self):
        v1 = [0, 0]
        v2 = [3, 4]
        # Euclidean (p=2): sqrt((0-3)^2 + (0-4)^2) = sqrt(9+16) = 5
        dist = calculate_minkowski(v1, v2, 2)
        self.assertAlmostEqual(dist, 5.0)

    def test_minkowski_dimension_mismatch(self):
        v1 = [1, 2]
        v2 = [1, 2, 3]
        with self.assertRaises(ValueError):
            calculate_minkowski(v1, v2, 2)

if __name__ == '__main__':
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#Q3
import time
import random

# Generated using Gemini AI
def benchmark_kmeans(data, k, iterations=5):
    """
    Compares the execution time of two different K-means implementations.
    """
    print(f"Benchmarking K-Means with K={k} over {iterations} runs...")
    
    # Time AI version
    ai_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        # Replace with the AI generated function
        kmeans_clustering(data, k) 
        ai_times.append(time.perf_counter() - start_time)
        
    # Time Own Version (Placeholder for your custom implementation)
    own_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        # Replace with your manual implementation: kmeans_clustering_own(data, k)
        kmeans_clustering(data, k) # Dummy call for illustration
        own_times.append(time.perf_counter() - start_time)
        
    avg_ai_time = sum(ai_times) / iterations
    avg_own_time = sum(own_times) / iterations
    
    print("-" * 30)
    print(f"AI Version Avg Time:  {avg_ai_time:.5f} seconds")
    print(f"Own Version Avg Time: {avg_own_time:.5f} seconds")
    print(f"Difference:           {abs(avg_ai_time - avg_own_time):.5f} seconds")

# Dummy data generation for the performance test
dummy_data = [[random.uniform(0, 100) for _ in range(5)] for _ in range(1000)]

# Run the performance comparison
benchmark_kmeans(dummy_data, k=3)

