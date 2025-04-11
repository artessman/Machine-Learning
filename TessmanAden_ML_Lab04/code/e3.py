
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


#exercise 3

full_X = np.array(df.drop(['MedHouseVal'], axis=1))
full_y= np.array(df['MedHouseVal'])

scaler = StandardScaler()
ScaledData = scaler.fit_transform(full_X)

reg = LinearRegression()
reg.fit(ScaledData,full_y)

coefficients = reg.coef_
max_coeff_index = abs(coefficients).argmax() #index of max coeffient(absolute value)

print(f'Most weighted Feature: {df.columns[max_coeff_index]}')

