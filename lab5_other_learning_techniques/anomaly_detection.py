from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot

# Create dataset
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.5)

# Isolation Forest
model = IsolationForest(contamination=0.05)
pred = model.fit_predict(X)

# Plot results
plot.scatter(X[:, 0], X[:, 1], c=pred)
plot.title("Anomaly Detection using Isolation Forest")
plot.savefig("./lab5_other_learning_techniques/fig_01_anomaly_detection.png")
print("Saved as \"fig_01_anomaly_detection.png\"")
plot.show()