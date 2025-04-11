
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2)

reg = LinearRegression()

data = pd.read_csv(r"C:\Users\Yeyian PC\source\repos\MLCumulativeNotesandAssignemnts\MLCumulativeNotesandAssignemnts\HWs\HomeWork4\avgHigh_jan_1895-2018.csv")
pred_point = np.array([201901, 202301, 202401]).reshape(-1, 1)


X = np.array(data.iloc[:,0]).reshape(-1, 1)
y = np.array(data.iloc[:,1]).reshape(-1, 1)

reg.fit(X, y)

plt.scatter(X, y, color='b', label = 'datapoints')
plt.plot(X, reg.predict(X), color='r', label= 'Model')
plt.scatter(pred_point, reg.predict(pred_point), color='g', label= 'Predicted')
plt.legend(loc='lower right')
plt.title(f'Jan Average High Temps. Slope: {reg.coef_} Intercept: {reg.intercept_}')
plt.ylabel('Temperatures')
plt.xlabel('Year')
plt.show()
