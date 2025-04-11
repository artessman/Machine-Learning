
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#pip install opencv-python
import cv2 #use the install command above in: View->Other Windows->Python Environments->Packages (PyPI)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

#uncomment if using google colab:
#from google.colab.patches import cv2_imshow

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

train_img = train.drop('label', axis=1)
train_lbl = train['label']
test_img = test.drop('label', axis=1)
test_lbl = test['label']


logisticRegr = LogisticRegression(solver = 'lbfgs', multi_class='multinomial')

logisticRegr.fit(train_img, train_lbl)
#if predict a single image then reshape:
#Ypred = logisticRegr.predict(test_img[0].reshape(1, -1))

predictions = logisticRegr.predict(test_img)
score = logisticRegr.score(test_img, test_lbl)
#print(score)

confusionMatrix = confusion_matrix(test_lbl, predictions)
print(f'Confusion Matrix:\n{confusionMatrix}')
print(f'Accuracy score: {accuracy_score(test_lbl, predictions)}')
print(f'Classification Report:\n{classification_report(test_lbl, predictions)}')

disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
display_labels=logisticRegr.classes_)
disp.plot()
plt.show()
