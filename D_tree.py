import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_split=2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.tree = None

    def gini(self, y):
        #Calculate Gini impurity
        counts = Counter(y)
        impurity = 1
        for i, count in counts.items():
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity

    def best_split(self, X, y):
        #the best split for the data
        best_feat, best_thresh, best_gain = None, None, -1
        n_samples, n_features = X.shape
        parent_impurity = self.gini(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue

                left_y, right_y = y[left_idx], y[right_idx]
                impurity = (len(left_y)/n_samples)*self.gini(left_y) + \
                           (len(right_y)/n_samples)*self.gini(right_y)
                gain = parent_impurity - impurity

                if gain > best_gain:
                    best_feat, best_thresh, best_gain = feat, thresh, gain

        return best_feat, best_thresh, best_gain

    def build_tree(self, X, y, depth=0):
       #Recursively build the decision tree.
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_split or \
           num_labels == 1:
            return Counter(y).most_common(1)[0][0]

        feat, thresh, gain = self.best_split(X, y)
        if gain == -1:
            return Counter(y).most_common(1)[0][0]

        left_idx = X[:, feat] <= thresh
        right_idx = ~left_idx

        left_br = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right_br = self.build_tree(X[right_idx], y[right_idx], depth+1)

        return (feat, thresh, left_br, right_br)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feat, thresh, left, right = tree
        if x[feat] <= thresh:
            return self.predict_one(x, left)
        else:
            return self.predict_one(x, right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
