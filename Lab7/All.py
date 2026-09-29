import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter
from math import log2
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

DB_DIR = Path(__file__).parent / "ICBHI_final_database"


def load_icbhi_dataset():
    loc_map    = {"Al": 0, "Ar": 1, "Ll": 2, "Lr": 3,
                  "Pl": 4, "Pr": 5, "Tc": 6}
    mode_map   = {"sc": 0, "mc": 1}
    device_map = {"Meditron": 0, "Litt3200": 1, "LittC2SE": 2, "AKGC417L": 3}

    records = []
    for fpath in glob.glob(str(DB_DIR / "*.txt")):
        stem  = Path(fpath).stem
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        patient_id  = int(parts[0])
        location    = parts[2]
        mode        = parts[3]
        device      = parts[4]

        loc_code    = loc_map.get(location, -1)
        mode_code   = mode_map.get(mode, -1)
        device_code = device_map.get(device, -1)

        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    continue
                try:
                    start   = float(cols[0])
                    end     = float(cols[1])
                    crackle = int(cols[2])
                    wheeze  = int(cols[3])
                except ValueError:
                    continue

                duration = end - start
                label    = crackle * 2 + wheeze
                label    = {0: 0, 2: 1, 1: 2, 3: 3}[label]

                records.append({
                    "patient_id": patient_id,
                    "location"  : loc_code,
                    "mode"      : mode_code,
                    "device"    : device_code,
                    "start"     : start,
                    "end"       : end,
                    "duration"  : duration,
                    "crackle"   : crackle,
                    "wheeze"    : wheeze,
                    "label"     : label,
                })

    return pd.DataFrame(records)


def equal_width_bin(values, bins=4):
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if lo == hi:
        return np.zeros(len(values), dtype=int)
    edges      = np.linspace(lo, hi, bins + 1)
    edges[-1] += 1e-9
    return np.digitize(values, edges) - 1


def calculate_entropy(values, bins=4):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        values = equal_width_bin(values, bins)

    counts = Counter(values)
    n      = len(values)
    h      = 0.0
    for c in counts.values():
        p  = c / n
        if p > 0:
            h -= p * log2(p)
    return h


def calculate_gini(values):
    values = np.asarray(values)
    counts = Counter(values)
    n      = len(values)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def information_gain(feature_vals, target_vals, bins=4):
    feature_vals = np.asarray(feature_vals)
    target_vals  = np.asarray(target_vals)

    if feature_vals.dtype.kind == "f":
        feature_vals = equal_width_bin(feature_vals, bins)

    h_parent = calculate_entropy(target_vals)
    n        = len(target_vals)
    h_cond   = 0.0
    for val in np.unique(feature_vals):
        mask    = feature_vals == val
        h_cond += (mask.sum() / n) * calculate_entropy(target_vals[mask])
    return h_parent - h_cond


def select_root_feature(features, target, bins=4):
    target     = np.asarray(target)
    best_feat, best_ig = None, -1.0
    for col in features.columns:
        ig = information_gain(features[col].values, target, bins=bins)
        if ig > best_ig:
            best_ig   = ig
            best_feat = col
    return best_feat, best_ig


def equal_frequency_bin(values, bins=4):
    values    = np.asarray(values, dtype=float)
    quantiles = np.linspace(0, 100, bins + 1)
    edges     = np.percentile(values, quantiles)
    edges[-1] += 1e-9
    _, idx = np.unique(edges, return_index=True)
    if len(idx) < 2:
        return np.zeros(len(values), dtype=int)
    return np.digitize(values, edges[1:])


def bin_feature(values, method="equal_width", bins=4):
    if method == "equal_width":
        return equal_width_bin(values, bins)
    elif method == "equal_frequency":
        return equal_frequency_bin(values, bins)
    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         "Use 'equal_width' or 'equal_frequency'.")


class _DecisionNode:
    def __init__(self, feature_idx=None, children=None, label=None, ig=0.0):
        self.feature_idx = feature_idx
        self.children    = children
        self.label       = label
        self.ig          = ig


class CustomDecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2,
                 binning_method="equal_width", bins=4):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.binning_method    = binning_method
        self.bins              = bins
        self.root_             = None
        self.feature_names_    = None

    def _bin_col(self, col):
        return bin_feature(col, method=self.binning_method, bins=self.bins)

    def _majority(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _ig(self, col_binned, y):
        n        = len(y)
        h_parent = calculate_entropy(y)
        h_cond   = 0.0
        for bv in np.unique(col_binned):
            mask    = col_binned == bv
            h_cond += (mask.sum() / n) * calculate_entropy(y[mask])
        return h_parent - h_cond

    def _best_split(self, X_np, y):
        best_idx, best_ig_val, best_binned = 0, -1.0, None
        for j in range(X_np.shape[1]):
            binned = self._bin_col(X_np[:, j])
            ig_val = self._ig(binned, y)
            if ig_val > best_ig_val:
                best_ig_val = ig_val
                best_idx    = j
                best_binned = binned
        return best_idx, best_ig_val, best_binned

    def _build(self, X_np: np.ndarray, y: np.ndarray, depth: int):
        if (len(np.unique(y)) == 1
                or len(y) < self.min_samples_split
                or X_np.shape[1] == 0
                or (self.max_depth is not None and depth >= self.max_depth)):
            return _DecisionNode(label=self._majority(y))

        best_idx, best_ig_val, best_binned = self._best_split(X_np, y)
        if best_ig_val <= 0:
            return _DecisionNode(label=self._majority(y))

        children = {}
        for bv in np.unique(best_binned):
            mask = best_binned == bv
            children[bv] = self._build(X_np[mask], y[mask], depth + 1)

        return _DecisionNode(feature_idx=best_idx, children=children,
                             ig=best_ig_val)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_np = X.values.astype(float)
        else:
            X_np = np.asarray(X, dtype=float)
            self.feature_names_ = [str(i) for i in range(X_np.shape[1])]
        y_np = np.asarray(y)
        self.root_ = self._build(X_np, y_np, depth=0)
        return self

    def _predict_row(self, row: np.ndarray, node: _DecisionNode):
        if node.label is not None:
            return node.label
        bv    = self._bin_col(row[node.feature_idx:node.feature_idx+1])[0]
        child = node.children.get(bv, next(iter(node.children.values())))
        return self._predict_row(row, child)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(float)
        else:
            X_np = np.asarray(X, dtype=float)
        return np.array([self._predict_row(X_np[i], self.root_)
                         for i in range(len(X_np))])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def print_tree(self, node=None, depth=0, label_names=None):
        if node is None:
            node = self.root_
        pad = "  " * depth
        if node.label is not None:
            name = (label_names[node.label] if label_names else str(node.label))
            print(f"{pad}>> Leaf: {name}")
        else:
            fname = (self.feature_names_[node.feature_idx]
                     if self.feature_names_ else str(node.feature_idx))
            print(f"{pad}[{fname}]  IG={node.ig:.4f}")
            for bv, child in node.children.items():
                print(f"{pad}  bin={bv}:")
                self.print_tree(child, depth + 2, label_names)


def build_decision_tree(features, target, binning_method="equal_width", bins=4):
    tree = CustomDecisionTree(binning_method=binning_method, bins=bins)
    tree.fit(features, target)
    return tree


def visualize_decision_tree(clf, feature_names=None,
                             class_names=None, max_depth=3,
                             title="Decision Tree"):
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(clf, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, max_depth=max_depth, ax=ax)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    out = Path(__file__).parent / "decision_tree.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[A6] Decision tree visualisation saved -> {out}")


def plot_decision_boundary(features, target, feature_names=None,
                            title="Decision Boundary"):
    X    = np.asarray(features)[:, :2]
    y    = np.asarray(target)
    feat = (list(feature_names)[:2] if feature_names is not None
            else ["Feature 0", "Feature 1"])

    clf  = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors  = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    classes = sorted(np.unique(y))
    cmap    = plt.cm.colors.ListedColormap([colors[c] for c in classes])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,
               edgecolors="k", linewidths=0.4, s=18, alpha=0.8)

    label_names = ["Normal", "Crackle", "Wheeze", "Both"]
    patches     = [mpatches.Patch(color=colors[c],
                                   label=label_names[c]) for c in classes]
    ax.legend(handles=patches, title="Class")
    ax.set_xlabel(feat[0]); ax.set_ylabel(feat[1])
    ax.set_title(title)
    plt.tight_layout()
    out = Path(__file__).parent / "decision_boundary.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[A7] Decision boundary saved -> {out}")


def tune_hyperparameters(features, target, search="grid", cv=5,
                          random_state=42):
    param_grid = {
        "criterion"        : ["gini", "entropy"],
        "max_depth"        : [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }

    base = DecisionTreeClassifier(random_state=random_state)

    if search == "grid":
        searcher = GridSearchCV(base, param_grid, cv=cv,
                                scoring="accuracy", n_jobs=-1, verbose=0)
        tag = "GridSearchCV"
    else:
        searcher = RandomizedSearchCV(base, param_grid, cv=cv,
                                      n_iter=20, scoring="accuracy",
                                      random_state=random_state,
                                      n_jobs=-1, verbose=0)
        tag = "RandomizedSearchCV"

    searcher.fit(features, target)
    print(f"\n[A8] {tag}")
    print(f"  Best params  : {searcher.best_params_}")
    print(f"  Best CV acc  : {searcher.best_score_:.4f}")
    return searcher.best_estimator_, searcher.best_params_, searcher.best_score_


if __name__ == "__main__":

    print("Loading ICBHI dataset …")
    df = load_icbhi_dataset()
    print(f"  Total breath cycles : {len(df)}")
    print("  Label distribution :")
    print(df["label"].value_counts().sort_index()
            .rename({0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}))
    print()

    LABEL_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]
    FEATURES    = ["duration", "location", "mode", "device", "patient_id"]
    X = df[FEATURES]
    y = df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    sep = "=" * 60
    print(sep); print("A1. ENTROPY"); print(sep)
    h_label    = calculate_entropy(y.values)
    h_duration = calculate_entropy(df["duration"].values, bins=4)
    print(f"  H(label)             = {h_label:.4f} bits")
    print(f"  H(duration, 4 bins)  = {h_duration:.4f} bits\n")

    print(sep); print("A2. GINI INDEX"); print(sep)
    gini = calculate_gini(y.values)
    print(f"  Gini(label) = {gini:.4f}\n")

    print(sep); print("A3. ROOT FEATURE SELECTION (Information Gain)"); print(sep)
    for feat in FEATURES:
        ig = information_gain(df[feat].values, y.values, bins=4)
        print(f"  IG({feat:<12}) = {ig:.4f}")
    best_feat, best_ig = select_root_feature(X, y, bins=4)
    print(f"\n  >> Root node : '{best_feat}'  (IG = {best_ig:.4f})\n")

    print(sep); print("A4. BINNING DEMO (first 10 durations)"); print(sep)
    sample = df["duration"].values[:10]
    print(f"  Raw          : {np.round(sample, 3)}")
    print(f"  equal_width  : {bin_feature(sample, 'equal_width',     bins=4)}")
    print(f"  equal_freq   : {bin_feature(sample, 'equal_frequency', bins=4)}\n")

    print(sep); print("A5. CUSTOM DECISION TREE"); print(sep)
    rng = np.random.RandomState(42)
    sub_idx  = rng.choice(len(X_tr), size=min(600, len(X_tr)), replace=False)
    X_sub    = X_tr.iloc[sub_idx]
    y_sub    = y_tr.iloc[sub_idx]
    sub_te   = rng.choice(len(X_te), size=min(200, len(X_te)), replace=False)
    X_sub_te = X_te.iloc[sub_te]
    y_sub_te = y_te.iloc[sub_te]

    print("  Fitting CustomDecisionTree on 600-sample subsample ...")
    cdt = CustomDecisionTree(max_depth=4, binning_method="equal_width", bins=4)
    cdt.fit(X_sub, y_sub)
    print(f"  Custom DT accuracy (subsample test): {cdt.score(X_sub_te, y_sub_te):.4f}")
    print("\n  Tree structure (depth<=4):")
    cdt.print_tree(label_names=LABEL_NAMES)
    print()

    print(sep); print("A5/A6. SKLEARN DECISION TREE"); print(sep)
    sk_clf = DecisionTreeClassifier(criterion="entropy", max_depth=7,
                                     random_state=42)
    sk_clf.fit(X_tr, y_tr)
    y_pred = sk_clf.predict(X_te)
    print(f"  sklearn DT accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=LABEL_NAMES))

    print(sep); print("A6. DECISION TREE VISUALISATION"); print(sep)
    visualize_decision_tree(sk_clf,
                             feature_names=FEATURES,
                             class_names=LABEL_NAMES,
                             max_depth=3,
                             title="ICBHI - Decision Tree (depth 3)")

    print(sep); print("A7. DECISION BOUNDARY (duration vs location)"); print(sep)
    plot_decision_boundary(df[["duration", "location"]], y,
                            feature_names=["Duration (s)", "Chest Location"],
                            title="Decision Boundary – Duration vs Location")

    print(sep); print("A8. HYPER-PARAMETER TUNING"); print(sep)
    best_g, _, _ = tune_hyperparameters(X_tr, y_tr, search="grid",   cv=5)
    best_r, _, _ = tune_hyperparameters(X_tr, y_tr, search="random", cv=5)

    print(f"\n  GridSearchCV   test acc : {accuracy_score(y_te, best_g.predict(X_te)):.4f}")
    print(f"  RandomizedCV   test acc : {accuracy_score(y_te, best_r.predict(X_te)):.4f}")

    print("\n✓ All tasks A1–A8 complete. PNG outputs saved to Lab7/.")