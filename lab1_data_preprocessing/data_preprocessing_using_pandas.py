# ***INSTALL PACKAGES***
# pip install pandas

import pandas as pd

# Creating DataFrame
data = {
    'Age': [25, 30, 35, 40, None],
    'Salary': [50000, 60000, None, 80000, 90000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Handling missing values
df.fillna({'Age': df['Age'].mean()}, inplace=True)
df.fillna({'Salary': df['Salary'].mean()}, inplace=True)
print("\nAfter Filling Missing Values:\n", df)

# One-Hot Encoding categorical data
one_hot = pd.get_dummies(df['City'], prefix='City')
df_encoded = pd.concat([df, one_hot], axis= 1)
print("\nAfter Encoding Categorical Data:\n", df_encoded)