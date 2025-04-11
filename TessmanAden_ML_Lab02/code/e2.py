import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = np.array([[1, 5], 
                 [3, 2], 
                 [8, 4], 
                 [7, 14]])

def myMean(x):
    numSum = 0
    for i in range(len(x)):
        numSum += x[i]
    return numSum/len(x)

def myStd(x):
    mean = myMean(x)
    squared_differences = []
    for i in range(len(x)):  
        squared_diff = (x[i] - mean) ** 2 
        squared_differences.append(squared_diff)
    numSum = 0
    for sq_diff in squared_differences: 
        numSum += sq_diff
    variance = numSum / len(x)
    return variance ** 0.5

def myStand(x):
    row, col = x.shape
    copy = x.astype(np.float64)
    stds = []
    means = []
    for i in range(col):
        temp = np.copy(x[:, i]) 
        mean = myMean(temp)
        std = myStd(temp)
        for j in range(len(temp)):
            z = (temp[j] - mean) / std
            copy[j, i] = z
        means.append(mean)
        stds.append(std)
    return copy, stds, means


def myStandRev(x, stds, means):
    row, col = x.shape
    copy = np.copy(x)
    for i in range(col):
        temp = np.copy(x[:, i])  
        for j in range(len(temp)):
            z = (temp[j] * stds[i]) + means[i] 
            copy[j, i] = z
    return copy


# Built-in Standardization 
scaler = StandardScaler()
scaler.fit(data)
sklearn_standardized = scaler.transform(data)

# My standardization
my_standardized, stds, means = myStand(data)

print("Sklearn Standardized Data:")
print(sklearn_standardized)
print("\nMy Standardized Data:")
print(my_standardized)

# Built in destandardization 
sklearn_rev_standardized = scaler.inverse_transform(sklearn_standardized)

#custom destandardization
original =  myStandRev(my_standardized, stds, means)

print("\nSklearn Destandardized Data:")
print(sklearn_rev_standardized)
print("\nMy Destandardized Data:")
print(original)






