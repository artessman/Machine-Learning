
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('speedLimits.csv')

X =  np.array(data['Speed']).reshape(-1, 1)
y =  np.array(data['Ticket']).reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Define different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
classifiers = {}

# Train SVM models with different kernels
for kernel in kernels:
    clf = SVC(kernel=kernel, C=1.0)
    clf.fit(X_train, y_train)
    classifiers[kernel] = clf

# Evaluate the models
accuracies = {}
for kernel, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[kernel] = accuracy
    print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")

#print best
print(f'best Kernal: {max(accuracies.items(), key=lambda k: k[1])}')

#plot
for i in range(len(data)):
    if data.iloc[i, 1] == 'NT':
        plt.scatter(data.iloc[i,0],data.iloc[i,1], color = 'g')
    else:
        plt.scatter(data.iloc[i,0],data.iloc[i,1], color = 'r')

plt.title('Speed vs Ticket')
plt.xlabel('Speed')
plt.ylabel('Ticket')
plt.show()


