
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision = 2, suppress = True)

df = pd.read_csv('Bank-data.csv')
X = np.array(df.iloc[:, 1:7])
y = np.array(df['y'].map(lambda x: 1 if x == 'yes' else 0))
#scale Data
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

#preform logistic regression
logReg = LogisticRegression()
logReg.fit(Xscaled, y)

print('Logistic Regression Coefficients: ', logReg.coef_)
print('Logistic Regression Intercept: ', logReg.intercept_)

dataPoints = np.array([
    [1.335, 0, 1, 0, 0, 109], 
    [1.25, 0, 0, 1, 0, 279]
    ])

dataPoints_scaled = scaler.transform(dataPoints)
yPred = logReg.predict(dataPoints_scaled)

#.predict_proba() gives probabilities for each class: [P(y=0), P(y=1)]
yProb = logReg.predict_proba(dataPoints_scaled)[:, 1]

odds = np.exp(logReg.coef_)
print(f'Odds:{odds}')
for c in range(yPred.shape[0]):
    if yPred[c] == 1:
        print(f'Client {c+1} Predition: yes')
    elif yPred[c] == 0:
        print(f'Client {c+1} Prediction: no')
for c in range(yProb.shape[0]):
    print(f'Client {c+1} has a {yProb[c]*100:.2f}% chance of subscribing: ')
    

