from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

# Create Dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=0)

# K-Means Model
model = KMeans(n_clusters=3)
labels = model.fit_predict(X)

# Plot
plot.scatter(X[:, 0], X[:, 1], c=labels)
plot.title("K-Means Clustering")
plot.savefig("./lab3_unsupervised_learning/fig_01_k_means_clustering.png")
print("Saved as \"fig_01_k_means_clustering.png\"")
plot.show()