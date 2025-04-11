
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

names = [ "Sample code number", "Clump Thickness", "Uniformity of Cell Size",
    "Uniformity of Cell Shape", "Marginal Adhesion","Single Epithelial Cell Size",
    "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli","Mitoses","Class"
]
df = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)

print(len(df))

df.replace('?', pd.NA, inplace=True) 
df.dropna(inplace=True) 
df.reset_index(drop=True, inplace=True)
    
print(len(df))

X = np.array(df.loc[:, 'Clump Thickness':'Mitoses'])
y = np.array(df['Class'])

for i in range(len(y)):
    if y[i] == 2:
        y[i] = 0 #  benign 
    else:
        y[i] = 1 # mailignant


#use random_state=value to select the same data points in every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) #set seed to 42 all the time

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))

conf_matrix = confusion_matrix(y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
