
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

names = ['class', 'Alcohol','Malic Acid','Ash','Acadlinity','Magnisium','Total Phenols',
         'Flavanoids','NonFlavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline' ]
df = pd.read_csv('wine.data.csv', names=names)
X = np.array(df.iloc[:, 1:14])
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 )

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))
print('Index\tPredicted\tActual')
for i in range(len(pred)):
    if pred[i]!=y_test[i]:
        print(i, '\t', pred[i], '\t', y_test[i], '***')

print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, pred)}')

np.set_printoptions(precision = 2, suppress = True)
DataToPredict = np.array([[14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065],
                         [12.64,1.36,2.02,16.8,100,2.02,1.41,.53,.62,5.75,.98,1.59,450],
                         [12.53,5.51,2.64,25,96,1.79,.6,.63,1.1,5,.82,1.69,515],
                         [13.49,3.59,2.19,19.5,88,1.62,.48,.58,.88,5.7,.81,1.82,580]])

pred = knn.predict(DataToPredict)

print('Predicted Results\n')
for i in range(len(pred)):
    print('\t', DataToPredict[i], '\t', pred[i])
