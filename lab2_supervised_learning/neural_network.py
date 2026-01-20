from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

# Model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Neural Network Accuracy:", accuracy_score(y_test, y_pred))