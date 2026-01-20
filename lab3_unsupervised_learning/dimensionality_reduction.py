from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

# Create Dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=0)

# Reduce dimensions to 2
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot
plot.scatter(X_reduced[:, 0], X_reduced[:, 1])
plot.title("Dimensionality Reduction using PCA")
plot.savefig("./lab3_unsupervised_learning/fig_04_dimensionality_reduction.png")
print("Saved as \"fig_04_dimensionality_reduction.png\"")
plot.show()