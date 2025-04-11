
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision = 2, suppress = True)

df = pd.read_csv('Student-Pass-Fail.csv')
X = np.array(df.drop(['Pass_Or_Fail'], axis=1))
y = np.array(df['Pass_Or_Fail'])

#scale Data
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

#preform logistic regression
logReg = LogisticRegression()
logReg.fit(Xscaled, y)

print('Logistic Regression Coefficients: ', logReg.coef_)
print('Logistic Regression Intercept: ', logReg.intercept_)

dataPoints = np.array([
[7, 28],
[10, 34],
[2, 39]
])

dataPoints_scaled = scaler.transform(dataPoints)
yPred = logReg.predict(dataPoints_scaled)

#.predict_proba() gives probabilities for each class: [P(y=0), P(y=1)]
yProb = logReg.predict_proba(dataPoints_scaled)[:, 1]

odds = np.exp(logReg.coef_)
print('Odds of pass/fail')
print(odds)
#print(f'yProb: {yProb}')

for c in range(yPred.shape[0]):
    if yPred[c] == 1:
        print(f'Client {c+1} Predition: Pass')
    elif yPred[c] == 0:
        print(f'Client {c+1} Prediction: Fail')
for c in range(yProb.shape[0]):
    print(f'Student {c+1} probality of passing: {yProb[c]*100:.2f}%')
    

