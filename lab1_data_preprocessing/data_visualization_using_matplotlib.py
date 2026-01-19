# ***INSTALL PACKAGES***
# pip install matplotlib
# pip install scikit-learn

import pandas as pd
import matplotlib.pyplot as plot

df = pd.DataFrame({
    'Age': [25, 30, 35, 40, 32.5],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'City': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
})
plot.scatter(df['Age'], df['Salary'])
plot.xlabel("Age")
plot.ylabel("Salary")
plot.title("Age vs Salary")
plot.savefig("./lab1_data_preprocessing/fig_01_scatterplot.png")
print("Saved as \"fig_01_scatterplot.png\"")
plot.show()