import numpy as np

class CustomKNN:
    def __init__(self, k=3, weights='uniform', sort_algo='bubble'):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if weights not in ('uniform', 'distance'):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if sort_algo not in ('bubble', 'selection', 'insertion'):
            raise ValueError("sort_algo must be 'bubble', 'selection', or 'insertion'")
        self.k = k
        self.weights = weights
        self.sort_algo = sort_algo
        
    def _encode(self, y):
        classes = np.unique(y)
        self.class_map = {c: i for i, c in enumerate(classes)}
        self.reverse_map = {i: c for i, c in enumerate(classes)}
        return np.array([self.class_map[c] for c in y])
        
    def _impute(self, X):
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        return X
        
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("X must be a non-empty 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        self.X_train = self._impute(X)
        self.y_train = self._encode(y)
        self.n_features_in_ = self.X_train.shape[1]
        return self
        
    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
        
    def _sort(self, distances):
        # Includes Bubble, Selection, and Insertion sort
        arr = list(enumerate(distances))
        n = len(arr)
        
        if self.sort_algo == 'bubble':
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j][1] > arr[j+1][1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
        elif self.sort_algo == 'selection':
            for i in range(n):
                min_idx = i
                for j in range(i+1, n):
                    if arr[j][1] < arr[min_idx][1]:
                        min_idx = j
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
        elif self.sort_algo == 'insertion':
            for i in range(1, n):
                key = arr[i]
                j = i-1
                while j >= 0 and key[1] < arr[j][1]:
                    arr[j+1] = arr[j]
                    j -= 1
                arr[j+1] = key
        return arr
        
    def predict(self, X):
        X = self._impute(np.array(X, dtype=float))
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        predictions = []
        for x in X:
            dists = [self._distance(x, x_train) for x_train in self.X_train]
            neighbors = self._sort(dists)[:min(self.k, len(self.X_train))]
            
            votes = {}
            for idx, dist in neighbors:
                label = self.y_train[idx]
                weight = 1.0 if self.weights == 'uniform' else 1.0 / (dist + 1e-5)
                votes[label] = votes.get(label, 0) + weight
                
            pred_encoded = max(votes, key=votes.get)
            predictions.append(self.reverse_map[pred_encoded])
            
        return np.array(predictions)
        
    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return np.mean(preds == np.array(y_test))


import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def benchmark_model(model, X_train, X_test, y_train, y_test, name, runs=10):
    model.fit(X_train, y_train)
    
    times = []
    for _ in range(runs):
        start = time.time()
        preds = model.predict(X_test)
        times.append(time.time() - start)
        
    preds = model.predict(X_test) 
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, preds, average='weighted'):.4f}")
    print(f"F-score: {f1_score(y_test, preds, average='weighted'):.4f}")
    print(f"Avg Time ({runs} runs): {np.mean(times):.6f} seconds\n")

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    benchmark_model(CustomKNN(k=3), X_train, X_test, y_train, y_test,
                    "Custom KNN (Lab 5)")
    benchmark_model(KNeighborsClassifier(n_neighbors=3), X_train, X_test,
                    y_train, y_test, "Scikit-Learn KNN")
    benchmark_model(CustomKNN(k=3, weights='distance'), X_train, X_test,
                    y_train, y_test, "Custom Weighted KNN")