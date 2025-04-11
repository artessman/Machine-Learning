
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=2)

reg = LinearRegression()

data = pd.read_csv(r"C:\Users\Yeyian PC\source\repos\MLCumulativeNotesandAssignemnts\MLCumulativeNotesandAssignemnts\HWs\HomeWork4\avgHigh_jan_1895-2018.csv")
pred_point = np.array([201901, 202301, 202401]).reshape(-1, 1)


X = np.array(data.iloc[:,0]).reshape(-1, 1)
y = np.array(data.iloc[:,1]).reshape(-1, 1)

userin = float(input('Enter Test Size (0.xx): '))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=userin, shuffle=False)

reg.fit(X_train, y_train)
predicted = np.array(reg.predict(X_test)).reshape(-1, 1)

slope, intercept, r, p, std_error = stats.linregress(X_train.flatten(), y_train.flatten())
print(f'Slope: {slope}, y-intercept: {intercept}, r: {r}, p-value: {p}, std error: {std_error}')

for i in range(len(X_test)):
    print(f'Actual: {y_test[i]}, Predicted: {predicted[i]}')




rmse = np.sqrt(np.mean((y_test - predicted) ** 2))


print(f'RMSE: {rmse}')

inter = float(reg.intercept_)
coeff = float(reg.coef_)


plt.scatter(X_train, y_train, color='b', label = 'train')
plt.plot(X_train, reg.predict(X_train), color='r', label= 'Model')
plt.scatter(X_test, y_test, color='g', label= 'Predicted')
plt.legend(loc='lower right')
plt.title(f'Slope: {coeff} Intercept: {inter}. Test size: {userin} ({len(X_test)}/{len(X)}), RMSE: {rmse:.2f}')
plt.ylabel('Temperatures')
plt.xlabel('Year')
plt.show()
