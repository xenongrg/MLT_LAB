from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

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
y_l1_pred = model_l1.predict(X_test)

# L2 Regularization (Ridge)
model_l2 = LogisticRegression(l1_ratio=0, max_iter=5000)
model_l2.fit(X_train, y_train)
y_l2_pred = model_l2.predict(X_test)

print("L1 Regularization (Lasso)")
print("Precision:", precision_score(y_test, y_pred=y_l1_pred))
print("Recall:", recall_score(y_test, y_pred= y_l1_pred))
print("F1 Score:", f1_score(y_test, y_pred=y_l1_pred))

print("\nL2 Regularization (Ridge)")
print("Precision:", precision_score(y_test, y_pred=y_l2_pred))
print("Recall:", recall_score(y_test, y_pred= y_l2_pred))
print("F1 Score:", f1_score(y_test, y_pred=y_l2_pred))