import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model

data = {
    'SquareFeet' :[100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600,
                634, 718, 750, 850, 903, 978, 1010, 1050, 1990],
    'Price' : [12300, 18150, 20100, 23500, 31005,359000, 44359, 52000, 53853,
         61328, 68000, 72300, 77000, 89379, 93200, 97150, 102750, 115358, 119330, 323989]
    }

df = pd.DataFrame(data)

X = np.array(df['SquareFeet']).reshape(-1, 1)
y = np.array(df['Price'])

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
print(f"Before RANSAC: slope: {lr.coef_}, y-intercept: {lr.intercept_}")
print(f"After RANSAC: slope: {ransac.estimator_.coef_}, y-intercept: {ransac.estimator_.intercept_}")

plt.scatter(X[inlier_mask], y[inlier_mask], color="g", label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], color="r", label="Outliers")
plt.plot(line_X, line_y, color="b", linewidth=2, label="Before RANSAC")
plt.plot(line_X, line_y_ransac, color="orange", linewidth=2, label="After RANSAC")
plt.legend()
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Linear Regression Before and After RANSAC")
plt.show()
