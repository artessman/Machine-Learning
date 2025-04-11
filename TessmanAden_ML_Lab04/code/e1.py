
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

dp = np.array([8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]).reshape(1, -1)

reg = LinearRegression()
reg.fit(X,y)


# Getting the coefficients of the regression
for i in range(8):
    print(f'Coeff {i+1}: {reg.coef_[i]}')

pred = reg.predict(dp)
print(f'Predicted MedHouseVal: {pred}')