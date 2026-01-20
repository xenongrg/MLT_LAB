from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load Data
X, y = load_breast_cancer(return_X_y=True)

# Cross Validation
model = LogisticRegression(max_iter=5000)
scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:", scores)
print("Average Accuracy:", scores.mean())


# ***OUTPUT***
# Cross Validation Scores: [0.93859649 0.94736842 0.98245614 0.92982456 0.95575221]
# Average Accuracy: 0.9507995652848935