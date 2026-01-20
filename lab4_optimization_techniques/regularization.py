from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3
)

# L1 Regularization (Lasso)
model_l1 = LogisticRegression(l1_ratio=0, solver='liblinear', max_iter=5000)
model_l1.fit(X_train, y_train)

# L2 Regularization (Ridge)
model_l2 = LogisticRegression(l1_ratio=0, max_iter=5000)
model_l2.fit(X_train, y_train)

print("L1 Regularization Accuracy:", model_l1.score(X_test, y_test))
print("L2 Regularization Accuracy:", model_l2.score(X_test, y_test))