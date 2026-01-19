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
print("\nAfter One-Hot Encoding Categorical Data:\n", df_encoded)


# ***OUTPUT***
# Original DataFrame:
#      Age   Salary     City
# 0  25.0  50000.0    Delhi
# 1  30.0  60000.0   Mumbai
# 2  35.0      NaN    Delhi
# 3  40.0  80000.0  Chennai
# 4   NaN  90000.0   Mumbai

# After Filling Missing Values:
#      Age   Salary     City
# 0  25.0  50000.0    Delhi
# 1  30.0  60000.0   Mumbai
# 2  35.0  70000.0    Delhi
# 3  40.0  80000.0  Chennai
# 4  32.5  90000.0   Mumbai

# After One-Hot Encoding Categorical Data:
#      Age   Salary     City  City_Chennai  City_Delhi  City_Mumbai
# 0  25.0  50000.0    Delhi         False        True        False
# 1  30.0  60000.0   Mumbai         False       False         True
# 2  35.0  70000.0    Delhi         False        True        False
# 3  40.0  80000.0  Chennai          True       False        False
# 4  32.5  90000.0   Mumbai         False       False         True