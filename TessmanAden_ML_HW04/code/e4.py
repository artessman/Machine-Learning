
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=2)

reg = LinearRegression()

data = pd.read_csv(r"C:\Users\Yeyian PC\source\repos\MLCumulativeNotesandAssignemnts\MLCumulativeNotesandAssignemnts\HWs\HomeWork4\materials.csv")
pred_point = np.array([[32.1, 37.5, 128.95],
                        [36.9, 35.37, 130.03]])

X = np.array(data.loc[:,["Pressure","Temperature"]])
Pressure = X[:, 0]
Temperature = X[:, 1]

y = np.array(data.loc[:,"Strength"])

reg.fit(X,y)


X1, X2 = np.meshgrid(Pressure, Temperature)

Z = reg.intercept_ + reg.coef_[0]*X1 + reg.coef_[1]*X2

#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(Pressure, Temperature, y, c=y, cmap='Greens')
ax.set_title('3D Graph')
ax.set_xlabel('Pressure')
ax.set_zlabel('Strength')
ax.set_ylabel('Temperature')
plt.show()
