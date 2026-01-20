import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-Item Rating Matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4]
])

# Compute Similarity
similarity = cosine_similarity(ratings)

print("User Similarity Matrix:\n", similarity)