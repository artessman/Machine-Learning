import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.csv', names=names)
X = np.array(df.iloc[:, 0:4])
y = np.array(df['class'])
for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    elif y[i] == 'Iris-versicolor':
        y[i] = 2
    else:
        y[i] = 3

figure, axis = plt.subplots(1, 2)

axis[0].scatter(df['sepal_length'], df["sepal_width"], c=y)
axis[0].set_xlabel("Sepal Length")
axis[0].set_ylabel("Sepal Width")
axis[0].set_title("Figure 1: Sepal features")

axis[1].scatter(df['petal_length'], df["petal_width"], c=y)
axis[1].set_xlabel("Petal Length")
axis[1].set_ylabel("Petal Width")
axis[1].set_title("Figure 2: Petal features")

plt.show()