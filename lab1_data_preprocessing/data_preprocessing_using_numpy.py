# ***INSTALL PACKAGES***
# pip install numpy

import numpy as np

# Creating dataset
data = np.array([
    [25, 50000],
    [30, 60000],
    [35, np.nan],
    [40, 80000],
    [np.nan, 90000]
])
print("Original Data:\n", data)

# Handling missing values using mean
mean_values = np.nanmean(data, axis=0)
indices = np.where(np.isnan(data))
data[indices] = np.take(mean_values, indices[1])
print("\nAfter Handling Missing Values:\n", data)

# Normalization
normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
print("\nNormalized Data:\n", normalized_data)