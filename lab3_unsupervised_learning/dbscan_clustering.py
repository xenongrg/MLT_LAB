from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plot

# Create Dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=0)

# DBSCAN Clustering
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# Plot
plot.scatter(X[:, 0], X[:, 1], c=labels)
plot.title("DBSCAN Clustering")
plot.savefig("./lab3_unsupervised_learning/fig_03_dbscan_clustering.png")
print("Saved as \"fig_03_dbscan_clustering.png\"")
plot.show()