import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Lab5.a1 import CustomKNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
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
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    benchmark_model(CustomKNN(k=3), X_train, X_test, y_train, y_test,
                    "Custom KNN (Lab 5)")
    benchmark_model(KNeighborsClassifier(n_neighbors=3), X_train, X_test,
                    y_train, y_test, "Scikit-Learn KNN")