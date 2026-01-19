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