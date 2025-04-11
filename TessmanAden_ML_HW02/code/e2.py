from random import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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

columns = ['gender','ses','schtyp', 'read', 'write','math','science', 'socst','honors', 'awards']
X = np.array(df.loc[:, columns])
y = np.array(df['prog'])
y = y.astype('int')

print(f'Before Standrdization: {X}')
X = StandardScaler().fit_transform(X)
print(f'After Standrdization: {X}')
X = StandardScaler().fit_transform(X)
print('Standard')
print(X)

x= np.arange(1,11)

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
print(f'\nVariance Ratio: {explained_variance}')

plt.plot(x, np.cumsum(explained_variance))
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Ratio')
plt.title('PC= 1-10')
plt.xticks(range(1,11))
plt.show()

