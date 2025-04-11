
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('vehicles.csv')
X= np.array(data.loc[:, 'cyl':])
y = np.array(data.loc[:, 'mpg'])
data = data.drop(['make', 'mpg'], axis =1) #droping colums make and mpg from data so their 
                                            #column names are not included in the data.columns that is called later

#print(X)

xScaledData = StandardScaler()
xScaled = xScaledData.fit_transform(X)

reg = LinearRegression().fit(xScaled, y)#between scaledX and y

print(f'Weighted Coeff (scaled X): {reg.coef_}')
print(f'Intercept: {reg.intercept_}')

temp = abs(reg.coef_)
fiveBest = data.columns[temp.argsort()[-5:]]
print(f'\nBest Coefficents: {fiveBest}')


dataPoint = xScaledData.transform([[6, 163, 111, 3.9, 2.77, 16.45, 0, 1, 4, 4]])
pred = reg.predict(dataPoint)
