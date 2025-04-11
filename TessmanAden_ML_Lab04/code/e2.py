
from distutils.errors import CCompilerError, CompileError
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

california_housing = fetch_california_housing(as_frame=True)

df = california_housing.frame
df2= df[::10]

X = np.array(df2.drop(['MedHouseVal'], axis=1))
y= np.array(df2['MedHouseVal'])

# Exercise2

X2 = X[:, :2]

reg = LinearRegression()
reg.fit(X2,y)

b1 = reg.coef_[0]
b2 = reg.coef_[1]

X_1, X_2 = np.meshgrid(X2[:, 0], X2[:, 1])
print(f'X1:\n{X_1}')
print(f'X2:\n{X_2}')
Z = reg.intercept_ + b1*X_1 + b2*X_2
#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X_1, X_2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(X2[:, 0], X2[:, 1], y, c=y, cmap='Greens')
ax.set_title('3D Graph')
plt.show()
