
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


X = np.array(data.loc[:,"Time":"Temperature"])
y = np.array(data.loc[:,"Strength"])

slope, intercept, r, p, std_error = stats.linregress(X[:,0], y)
print('Time vs Strength')
print(f'Slope: {slope}, r: {r}')

slope, intercept, r, p, std_error = stats.linregress(X[:,1], y)
print('Time vs Strength')
print(f'Slope: {slope}, r: {r}')

slope, intercept, r, p, std_error = stats.linregress(X[:,2], y)
print('Temperature vs Strength')
print(f'Slope: {slope}, r: {r}')


reg.fit(X,y)


for i in range(2):
    Ymodel = reg.intercept_
    for j in range(3):
        Ymodel = Ymodel + reg.coef_[j]*pred_point[i,j]
    print(f'Prediction for {pred_point[i,:]}: {Ymodel}')