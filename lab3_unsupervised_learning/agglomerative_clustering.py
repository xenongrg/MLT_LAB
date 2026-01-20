from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plot

# Create Dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=0)

# Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

# Plot
plot.scatter(X[:, 0], X[:, 1], c=labels)
plot.title("Agglomerative Clustering")
plot.savefig("./lab3_unsupervised_learning/fig_02_agglomerative_clustering.png")
print("Saved as \"fig_02_agglomerative_clustering.png\"")
plot.show()