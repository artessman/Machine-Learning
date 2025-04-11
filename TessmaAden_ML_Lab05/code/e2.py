import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def myConfMatrix(y_Test, yPred):
    uniqueLabels = np.unique(y_Test)
    
    confMatrix = np.zeros((len(uniqueLabels), len(uniqueLabels)))
    for i in range(len(uniqueLabels)):
        for j in range(len(uniqueLabels)):
            #goes through each of the predicted labels and find the sum of TP FP FN TN
            confMatrix[i, j] = np.sum((y_Test == uniqueLabels[i]) & (yPred == uniqueLabels[j]))
    
    return confMatrix

def MyAccuracy(y_Test, yPred):
    right_preds = 0
    right_preds = np.sum(y_Test == yPred)
    accuracy_score =  (right_preds/len(y_Test))
    return accuracy_score


split = float(input('Enter Test Percentage (0.xx): '))

np.set_printoptions(precision = 2, suppress = True)


df = pd.read_csv('Student-Pass-Fail.csv')
X = np.array(df.drop(['Pass_Or_Fail'], axis=1))
y = np.array(df['Pass_Or_Fail'])

#scale Data
scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

cut = round(X.shape[0]*split)

X_Test = Xscaled[:cut, :]
y_Test = y[:cut]
X_Train = Xscaled[cut:, :]
y_Train = y[cut:]


#preform logistic regression
logReg = LogisticRegression()
logReg.fit(X_Train, y_Train)

print('Logistic Regression Coefficients: ', logReg.coef_)
print('Logistic Regression Intercept: ', logReg.intercept_)

yPred = logReg.predict(X_Test)


print('Model accuracy score: ', accuracy_score(y_Test, yPred))
print('My accuracy score: ', MyAccuracy(y_Test, yPred))



conf_matrix = confusion_matrix(y_Test, yPred)
print(f'\nConfusion Matrix: \n{conf_matrix}')
my_conf_matrix = myConfMatrix(y_Test, yPred)
print(f'\nMy Confusion Matrix: \n{my_conf_matrix}')


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=logReg.classes_, yticklabels=logReg.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


