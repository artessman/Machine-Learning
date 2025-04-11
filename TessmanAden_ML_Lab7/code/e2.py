import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('golf.csv', dtype=str)

X = np.array(df.iloc[:, :4])
y = np.array(df.iloc[:,4]) 
dataToPred = [ ['Rainy', 'Hot', 'High', 'TRUE'],
              ['Sunny', 'Mild', 'Normal', 'FALSE'],
              ['Sunny', 'Cool', 'High', 'FALSE']]

X = np.vstack((X, dataToPred))
#convert to numerals
le = preprocessing.LabelEncoder()
row, cols = X.shape
X_encoded = np.ones([row, cols])
for i in range(cols):
    X_encoded[:, i] = le.fit_transform(X[:, i])
  
dataToPred = X_encoded[-3:,:]
X_encoded = X_encoded[:-3,:]

#y_encoded = np.array(le.fit_transform(y))

model = GaussianNB()
model.fit(X_encoded, y)
y_pred = model.predict(dataToPred)
for i in range(len(y_pred)):
    print(f"prediction of Datapoint[{i+1}]: {y_pred[i]}")