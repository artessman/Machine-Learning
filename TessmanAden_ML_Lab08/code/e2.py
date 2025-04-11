import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

# Suppress all warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('breast-cancer-wisconsin-data.csv', header=None)
#clean data
data.replace('?', None, inplace = True)
data = data.dropna()

X =  np.array(data.iloc[:, 1:10])
print(X.shape)
y =  np.array(data.iloc[:,10]).reshape(-1, 1)

#standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

#PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principalcomponent 1', 'principalcomponent 2'])
principalDf['Classes'] = y

X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size=0.25, random_state=42)

clf = SVC(kernel='linear')  # Soft-margin with default C=1.0
clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of model: {accuracy}')
print(f'\nConfusion Matrix: \n{confusion_matrix(y_test, y_pred)}')

print(f'coefficents: {clf.coef_}')
yModel = -1*((clf.coef_[0,0]*principalDf['principalcomponent 1'] + clf.intercept_) /clf.coef_[0, 1])

#plot

class1 =  principalDf[principalDf['Classes'] == 2]
class2 =  principalDf[principalDf['Classes'] == 4]

plt.scatter(class1['principalcomponent 1'], class1['principalcomponent 2'], color = 'purple', label ='2')
plt.scatter(class2['principalcomponent 1'], class2['principalcomponent 2'], color = 'y', label ='4')
plt.plot(principalDf['principalcomponent 1'], yModel, c = 'g', label='boundary')
plt.legend(loc = 'lower left')
plt.ylim(min(principalDf['principalcomponent 2']), max(principalDf['principalcomponent 2']))
plt.title('SVC with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.show()
