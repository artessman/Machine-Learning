
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MyPCA(X, n_comp):
    #1. Scale Data (already done for me)
    #2. Find Covariance Matrix
    covMatrix = np.cov(X.T)

    #3. Compute Eigenvalues and Eigenvectors
    #https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    eVals, eVecs  = np.linalg.eig(covMatrix)

    #4. Sort Eigenvectors by Eigenvalues
    # [::-1] Sort from largest to smallest
    indices = np.argsort(eVals)[::-1][:n_comp]
    eVals, eVecs = eVals[indices], eVecs[:, indices]

    #5 Multiply Row Feature Vector by Data Pointsb To recover data:
    FData = np.matmul(X, eVecs)

    return FData
def MyScaler(X):
    row, col =  X.shape
    for i in range(col):
        mean = np.mean(X[:, i])
        std = np.std(X[:,i])
        for j in range(row): 
            X[j,i] =  (X[j,i] - mean)/std

    return X

data = np.array(pd.read_csv('materials.csv'))
print(data[:5])
scalledData = MyScaler(data)
print(scalledData[:5])

PCA_data = MyPCA(scalledData, n_comp = 2)
print(PCA_data)

plt.scatter(PCA_data[:,0], PCA_data[:,1], color ='b', label='PCA Data')
plt.legend()
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.title('My PCA = 2')
plt.show()