import numpy as np
from sklearn.linear_model import LinearRegression

# Time-series data
time = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
values = np.array([100, 120, 130, 150, 170])

# Model
model = LinearRegression()
model.fit(time, values)

# Predict future value
future_time = np.array([[6]])
prediction = model.predict(future_time)

print("Predicted Value at Time 6:", prediction[0])