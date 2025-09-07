import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Random_forest import RandomForest   # make sure your RandomForest file is named Random_forest.py

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize Random Forest
rf = RandomForest(n_estimators=10, max_depth=5, min_split=2)

# Fit model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
