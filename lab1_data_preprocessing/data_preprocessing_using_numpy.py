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


# ***OUTPUT***
# Original Data:
#  [[2.5e+01 5.0e+04]
#  [3.0e+01 6.0e+04]
#  [3.5e+01     nan]
#  [4.0e+01 8.0e+04]
#  [    nan 9.0e+04]]

# After Handling Missing Values:
#  [[2.50e+01 5.00e+04]
#  [3.00e+01 6.00e+04]
#  [3.50e+01 7.00e+04]
#  [4.00e+01 8.00e+04]
#  [3.25e+01 9.00e+04]]

# Normalized Data:
#  [[0.         0.        ]
#  [0.33333333 0.25      ]
#  [0.66666667 0.5       ]
#  [1.         0.75      ]
#  [0.5        1.        ]]