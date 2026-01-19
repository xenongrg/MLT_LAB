import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    'Age': [25, 30, 35, 40, 32.5],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
})

# Label Encoding
le = LabelEncoder()
df['City_Label'] = le.fit_transform(df['City'])

# Splitting Dataset
x = df[['Age', 'Salary']]
y = df['City_Label']
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1
)
print("Training Data:\n", x_train)
print("\nTesting Data:\n", x_test)


# ***OUTPUT***
# Training Data:
#      Age  Salary
# 1  30.0   60000
# 4  32.5   90000
# 0  25.0   50000
# 3  40.0   80000

# Testing Data:
#      Age  Salary
# 2  35.0   70000