from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3
)

# Low Depth Tree (high bias)
model_bias = DecisionTreeClassifier(max_depth=2)
model_bias.fit(X_train, y_train)
pred_bias = model_bias.predict(X_test)

# High Depth Tree (high variance)
model_variance =  DecisionTreeClassifier()
model_variance.fit(X_train, y_train)
pred_variance = model_variance.predict(X_test)

print("High Bias Accuracy:", accuracy_score(y_test, pred_bias))
print("High Variance Accuracy:", accuracy_score(y_test, pred_variance))