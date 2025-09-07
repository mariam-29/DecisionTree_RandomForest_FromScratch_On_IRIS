
import numpy as np
import pandas as pd
from D_tree import DecisionTree
from collections import Counter
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_split=2, max_f=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_split = min_split
        self.max_f = max_f
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        if self.max_f is None:
            self.max_f = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, min_split=self.min_split)
            X_sample, y_sample = self.bootstrap_sample(X, y)

            # random feature subset
            feat_idxs = np.random.choice(n_features, self.max_f, replace=False)
            tree.fit(X_sample[:, feat_idxs], y_sample)

            # store tree with its feature subset
            self.trees.append((tree, feat_idxs))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X[:, feat_idxs]) for tree, feat_idxs in self.trees])
        # majority vote
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
