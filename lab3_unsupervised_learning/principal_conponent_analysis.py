from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Create Dataset
X, y = make_blobs(n_samples=300, centers=3, random_state=0)

# PCA
pca = PCA(n_components=2)
pca.fit_transform(X)
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)


# ***OUTPUT***
# Explained Variance Ratio:
# [0.61270781 0.38729219]