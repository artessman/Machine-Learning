from random import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('hsbdemo.csv')

print(df.head())

# coverting features to numeric values
df.loc[df['gender'] == 'male' , 'gender'] = 0
df.loc[df['gender'] == 'female' , 'gender'] = 1

df.loc[df['ses'] == 'low' , 'ses'] = 0
df.loc[df['ses'] == 'middle' , 'ses'] = 1
df.loc[df['ses'] == 'high' , 'ses'] = 2

df.loc[df['schtyp'] == 'public' , 'schtyp'] = 0
df.loc[df['schtyp'] == 'private' , 'schtyp'] = 1

df.loc[df['honors'] == 'not enrolled' , 'honors'] = 0
df.loc[df['honors'] == 'enrolled' , 'honors'] = 1

df.loc[df['prog'] == 'vocation' , 'prog'] = 0
df.loc[df['prog'] == 'general' , 'prog'] = 1
df.loc[df['prog'] == 'academic' , 'prog'] = 2

print(df.head())


X = np.array(df.loc[:, ['gender','ses','schtyp', 'read', 'write','math','science', 'socst','honors', 'awards']])
y = np.array(df['prog'])
y = y.astype('int')

#use random_state=value to select the same data points in every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=3) 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))
print('Index\tPredicted\t\tActual')
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        print(i, '\t', pred[i], '\t', y_test[i], '***')

conf_matrix = confusion_matrix(y_test, pred)

print(f'\nConfusion Matrix: \n{conf_matrix}')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()