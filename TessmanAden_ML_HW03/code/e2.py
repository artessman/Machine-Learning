
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import math

# Load dataset
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.csv', header=None, names=names)
df['Index'] = [x for x in range(len(df))]

y = np.array(df['class'])
df = df.drop(['class'], axis=1)
X = np.array(df)

np.set_printoptions(precision=2, suppress=True) # Suppress scientific notation


'''
1. we split the dataset into k number of subsets (known as folds)
2.then we perform training on the all the subsets but leave one(k-1) subset for the evaluation of the trained model.


In this method, we iterate k times with a different subset reserved for testing purpose each time. The values for k can include three, five, and ten,
with two of the most common being k = 5 and k = 10. In each iteration, one fold is for testing, and the remaining k-1
folds are for training. After testing, you calculate the average of the results. K-fold cross-validation is commonly used
and highly adaptable to a variety of data sets.
'''

# My K-Fold Cross Validation

def Myk_fold(n_folds, X, y):
    fold_size = math.floor(len(X) / n_folds)  # Size of each fold
    accuracies = []
    
    for i in range(n_folds):
        # take this itterations test set (eval)
        test_start = i * fold_size
        test_end = (i + 1) * fold_size 
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        # take the rest of the data to be the train data
        
        X_train = np.empty(0)
        X_train = np.concatenate([X[:test_start], X[test_end:]], axis=0)
        y_train = np.empty(0)
        y_train = np.concatenate([y[:test_start], y[test_end:]], axis=0)
        
        #preform KNN
        knn = KNeighborsClassifier(n_neighbors=9)
        knn.fit(X_train, y_train)
        #get accuracy
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy}")
    
    # Print the average accuracy across all folds
    print(f"Average Accuracy: {np.mean(accuracies)}")


Myk_fold(5, X, y)
