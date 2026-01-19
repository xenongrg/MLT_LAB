# ***INSTALL PACKAGES***
# pip install scikit-learn

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame({
    'Age': [25, 30, 35, 40, 32.5],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
})
print("Original DataFrame:\n", df)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Salary']])
print("\nStandardized Data:\n", scaled_data)


# ***OUTPUT***
# Original DataFrame:
#      Age  Salary     City
# 0  25.0   50000    Delhi
# 1  30.0   60000   Mumbai
# 2  35.0   70000    Delhi
# 3  40.0   80000  Chennai
# 4  32.5   90000   Mumbai

# Standardized Data:
#  [[-1.5        -1.41421356]
#  [-0.5        -0.70710678]
#  [ 0.5         0.        ]
#  [ 1.5         0.70710678]
#  [ 0.          1.41421356]]