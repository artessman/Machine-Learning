from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

np.set_printoptions(precision=2)

data = pd.read_csv(r"C:\Users\Yeyian PC\source\repos\MLCumulativeNotesandAssignemnts\MLCumulativeNotesandAssignemnts\HWs\HomeWork4\materialsOutliers.csv")

data2 = data.copy()

X = np.array(data.loc[:,"Time":"Temperature"])
y = np.array(data.loc[:,"Strength"])

reg = LinearRegression()
reg.fit(X,y)
print("Before RANSAC:")
print(f'Coeff: {reg.coef_}') #print all coefficients
print(f'Y-intercept: {reg.intercept_}') #print all coefficients
Rsquared = reg.score(X,y)
print(f'R2: {Rsquared}') #print all coefficients

cumulative_mask = np.array([])
for i in range(X.shape[1]):
    ransac = RANSACRegressor(random_state=0, residual_threshold=15, stop_probability=1.00)
    ransac.fit(y.reshape(-1,1), X[:,i])
    inlier_mask = ransac.inlier_mask_
    if i == 0:
        cumulative_mask = inlier_mask
    else:
        cumulative_mask = cumulative_mask & inlier_mask

   

#use the combined inlier mask to filter the rows in the original df
data2 = data2[cumulative_mask]

X = np.array(data2.loc[:,"Time":"Temperature"])
y = np.array(data2.loc[:,"Strength"])

reg = LinearRegression()
reg.fit(X,y)
print("\nAfter RANSAC:")
print(f'Coeff: {reg.coef_}') #print all coefficients
print(f'Y-intercept: {reg.intercept_}') #print all coefficients
Rsquared = reg.score(X,y)
print(f'R2: {Rsquared}') #print all coefficients
