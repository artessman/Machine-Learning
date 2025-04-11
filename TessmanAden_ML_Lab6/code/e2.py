

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 #use the install command above in: View->Other Windows->Python Environments->Packages (PyPI)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

labels_keys = ['T-shirt', 'Trouers', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-Boot']

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

train_img = train.drop('label', axis=1)
train_lbl = train['label']
test_img = test.drop('label', axis=1)
test_lbl = test['label']
bag = cv2.cvtColor(cv2.imread('bag.jpg'), cv2.COLOR_BGR2GRAY).reshape(1, 28 * 28)
trousers = cv2.cvtColor(cv2.imread('trousers.bmp'), cv2.COLOR_BGR2GRAY).reshape(1, 28 * 28)

logisticRegr = LogisticRegression(solver = 'lbfgs', multi_class='multinomial')
logisticRegr.fit(train_img, train_lbl)

print(f'Predicted img1: {labels_keys[int(logisticRegr.predict(bag))]}')
print(f'Predicted img2: {labels_keys[int(logisticRegr.predict(trousers))]}')


